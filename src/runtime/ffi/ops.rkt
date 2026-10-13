#lang racket/base
;; Eager operator implementations over ctensors, assembled from
;; three sources: stable-ABI kernels, generated per-backend shims
;; (aoti_torch_cpu_* / aoti_torch_aten_*), the custom shim (zl_*), plus
;; Racket-side stride math for the view family. Consumed by eager.rkt via
;; eager-run. Every ctensor produced here carries a finalizer, so
;; intermediates are collected by GC.

(require (for-syntax racket/base
                     syntax/parse)
         ffi/unsafe
         racket/list
         (rename-in "raw.rkt" [bytes->ctensor unowned-bytes->ctensor])
         "wrapper.rkt"
         (only-in "../../operators/matmul.rkt" matmul-plan))

(provide eager-available?
         eager-run
         bytes->ctensor
         ctensor-shape
         ctensor-dtype-sym
         ctensor-contiguous)

;; The owning variant of the raw constructor: the returned ctensor is
;; finalized like every other ctensor produced by this module.
(define (bytes->ctensor dtype shape bs)
  (own (unowned-bytes->ctensor dtype shape bs)))

(define (eager-available?)
  (and (libtorch-available?) (shim-available?)))

(define (own h)
  (register-finalizer h ctensor-delete!)
  h)

;; --- generated-shim kernels (exported by libtorch_cpu itself) ---

(define cpu-add
  (lazy-fn "aoti_torch_cpu_add_Tensor"
           (_fun _pointer
                 _pointer
                 _double
                 (out : (_ptr o _pointer))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'add code)
                   out))))

(define cpu-mul
  (lazy-fn "aoti_torch_cpu_mul_Tensor"
           (_fun _pointer
                 _pointer
                 (out : (_ptr o _pointer))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'mul code)
                   out))))

(define cpu-abs
  (lazy-fn "aoti_torch_cpu_abs"
           (_fun _pointer
                 (out : (_ptr o _pointer))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'abs code)
                   out))))

(define cpu-cat
  (lazy-fn "aoti_torch_cpu_cat"
           (_fun (_list i _pointer)
                 _int64
                 _int64
                 (out : (_ptr o _pointer))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'cat code)
                   out))))

(define cpu-mm-out
  (lazy-fn "aoti_torch_cpu_mm_out"
           (_fun _pointer _pointer _pointer -> (code : _int32) -> (check-code 'matmul code))))

(define cpu-bmm-out
  (lazy-fn "aoti_torch_cpu_bmm_out"
           (_fun _pointer _pointer _pointer -> (code : _int32) -> (check-code 'matmul code))))

(define aten-amax
  (lazy-fn "aoti_torch_aten_amax"
           (_fun _pointer
                 (_list i _int64)
                 _int64
                 _int32
                 (out : (_ptr o _pointer))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'amax code)
                   out))))

;; --- ctensor helpers ---

(define dtype-syms '(f32 f16 bf16 i64 i32 u8 bool))

(define code->sym #f)

(define (ctensor-dtype-sym h)
  (unless code->sym
    (set! code->sym
          (for/hasheqv ([s (in-list dtype-syms)])
            (values (dtype-code s) s))))
  (hash-ref code->sym (ctensor-dtype-code h)))

(define (ctensor-shape h)
  (ctensor-sizes h))

;; clone preserves layout for strided views, so a contiguous copy is
;; materialized explicitly: fresh contiguous storage + copy_.
(define (ctensor-contiguous h)
  (cond
    [(ctensor-contiguous? h) h]
    [else
     (define out (own (make-ctensor (ctensor-dtype-sym h) (ctensor-shape h))))
     (ctensor-copy! out h)
     out]))

(define (view h sizes strides offset)
  (own (ctensor-reinterpret h sizes strides offset)))

(define (ctensor-reshape h shape)
  (define hc (ctensor-contiguous h))
  (view hc shape (shape->strides shape) (ctensor-storage-offset hc)))

(define missing-attr (gensym 'missing))

(define (aref attrs key [default missing-attr])
  (define p (assq key attrs))
  (cond
    [p (cdr p)]
    [(eq? default missing-attr)
     (raise-arguments-error 'eager "missing attribute" "key" key "attrs" attrs)]
    [else default]))

(define (norm-dim who rank d)
  (define nd
    (if (negative? d)
        (+ rank d)
        d))
  (unless (and (>= nd 0) (< nd rank))
    (raise-arguments-error who "dim out of range" "dim" d "rank" rank))
  nd)

(define (list-set* l i v)
  (for/list ([x (in-list l)]
             [j (in-naturals)])
    (if (= j i) v x)))

;; matmul: fold to a single mm/bmm call per matmul-plan, the same rank
;; analysis the shape rule uses.

(define (matmul-impl a b)
  (define sa (ctensor-shape a))
  (define sb (ctensor-shape b))
  (define-values (kind af bf of out) (matmul-plan sa sb))
  (define a*
    (if (equal? af sa)
        a
        (ctensor-reshape a af)))
  (define b*
    (if (equal? bf sb)
        b
        (ctensor-reshape b bf)))
  (define o (own (make-ctensor (ctensor-dtype-sym a) of)))
  (case kind
    [(mm) (cpu-mm-out o a* b*)]
    [(bmm) (cpu-bmm-out o a* b*)])
  (if (equal? of out)
      o
      (ctensor-reshape o out)))

;; --- the implementation table ---

(define impls (make-hasheq))

(begin-for-syntax
  (define-syntax-class impl-attr
    #:attributes (id default)
    (pattern id:id
      #:attr default #f)
    (pattern [id:id default:expr])))

;; The eager twin of define-operator: one entry per operator, the same
;; header shape minus the types. Expands to a table entry that
;; destructures the uniform (args attrs) calling convention into the named
;; operands and attributes around the body. The attribute key is the
;; parameter name, as on the registry side.
(define-syntax (define-operator-impl stx)
  (syntax-parse stx
    [(_ (name:id operand:id ... (~seq kw:keyword a:impl-attr) ...) body ...+)
     #'(hash-set! impls
                  'name
                  (lambda (args attrs)
                    (apply (lambda (operand ...)
                             (let ([a.id (aref attrs 'a.id (~? a.default))] ...)
                               body ...))
                           args)))]))

(define-operator-impl (add a b) (cpu-add a b 1.0))

(define-operator-impl (sub a b) (zl-sub a b))

(define-operator-impl (mul a b) (cpu-mul a b))

(define-operator-impl (div a b) (zl-div a b))

(define-operator-impl (neg x) (zl-neg x))

(define-operator-impl (abs x) (cpu-abs x))

(define-operator-impl (sqr x) (cpu-mul x x))

(define-operator-impl (rsqrt x) (zl-rsqrt x))

(define-operator-impl (silu x) (zl-silu x))

(define-operator-impl (cast x #:dtype dtype)
  (define out (own (make-ctensor dtype (ctensor-shape x))))
  (ctensor-copy! out x)
  out)

(define-operator-impl (mean x #:axis axis #:keep-dim? [keep-dim #f])
  (zl-mean x (list axis) 1 (if keep-dim 1 0)))

(define-operator-impl (amax x #:axis axis #:keep-dim? [keep-dim #f])
  (aten-amax x (list axis) 1 (if keep-dim 1 0)))

(define-operator-impl (argmax x #:axis axis #:keep-dim? [keep-dim #f])
  (zl-argmax x axis (if keep-dim 1 0)))

(define-operator-impl (softmax x #:axis axis) (zl-softmax x axis))

(define-operator-impl (embedding weight indices) (zl-embedding weight indices))

(define-operator-impl (matmul a b) (matmul-impl a b))

;; cat is variadic over its operands — outside the fixed-arity header
;; define-operator-impl destructures — so its entry stays hand-written.
(hash-set! impls 'cat (lambda (args attrs) (cpu-cat args (length args) (aref attrs 'axis))))

(define-operator-impl (reshape x #:shape shape) (ctensor-reshape x shape))

(define-operator-impl (transpose x #:dim0 dim0 #:dim1 dim1)
  (define sizes (ctensor-shape x))
  (define strides (ctensor-strides x))
  (define rank (length sizes))
  (define d0 (norm-dim 'transpose rank dim0))
  (define d1 (norm-dim 'transpose rank dim1))
  (view x
        (list-set* (list-set* sizes d0 (list-ref sizes d1)) d1 (list-ref sizes d0))
        (list-set* (list-set* strides d0 (list-ref strides d1)) d1 (list-ref strides d0))
        (ctensor-storage-offset x)))

(define-operator-impl (narrow x #:axis axis #:start start #:len len)
  (define sizes (ctensor-shape x))
  (define strides (ctensor-strides x))
  (define d (norm-dim 'narrow (length sizes) axis))
  (unless (<= 0 start (+ start len) (list-ref sizes d))
    (raise-arguments-error 'narrow
                           "slice out of range"
                           "start"
                           start
                           "len"
                           len
                           "size"
                           (list-ref sizes d)))
  (view x
        (list-set* sizes d len)
        strides
        (+ (ctensor-storage-offset x) (* start (list-ref strides d)))))

(define-operator-impl (broadcast-to x #:shape shape)
  (define sizes (ctensor-shape x))
  (define strides (ctensor-strides x))
  (define pad (- (length shape) (length sizes)))
  (unless (>= pad 0)
    (raise-arguments-error 'broadcast-to "target rank too small" "shape" sizes "target" shape))
  (define full-sizes (append (make-list pad 1) sizes))
  (define full-strides (append (make-list pad 0) strides))
  (define new-strides
    (for/list ([s (in-list full-sizes)]
               [st (in-list full-strides)]
               [t (in-list shape)])
      (cond
        [(= s t) st]
        [(= s 1) 0]
        [else
         (raise-arguments-error 'broadcast-to
                                "shape not broadcastable"
                                "shape"
                                sizes
                                "target"
                                shape)])))
  (view x shape new-strides (ctensor-storage-offset x)))

(define-operator-impl (unsqueeze x #:axis axis)
  (define sizes (ctensor-shape x))
  (define strides (ctensor-strides x))
  (define rank (length sizes))
  (define d (norm-dim 'unsqueeze (add1 rank) axis))
  (define new-stride
    (if (< d rank)
        (* (list-ref sizes d) (list-ref strides d))
        1))
  (view x
        (append (take sizes d) (list 1) (drop sizes d))
        (append (take strides d) (list new-stride) (drop strides d))
        (ctensor-storage-offset x)))

;; Scalar arguments are weak-dtyped: they enter the kernel in the dtype
;; the tensor operands agreed on, never as f64 tensors that would win
;; type promotion.
(define (as-ctensor v target)
  (cond
    [(real? v)
     (define h (own (real->ctensor v)))
     (cond
       [(or (not target) (eq? target 'f64)) h]
       [else
        (define out (own (make-ctensor target '())))
        (ctensor-copy! out h)
        out])]
    [else v]))

(define (eager-run name args attrs)
  (define f
    (hash-ref impls
              name
              (lambda () (raise-arguments-error 'eager "no eager implementation" "operator" name))))
  (define target
    (for/first ([a (in-list args)]
                #:unless (real? a))
      (ctensor-dtype-sym a)))
  (f (map (lambda (v) (as-ctensor v target)) args) attrs))
