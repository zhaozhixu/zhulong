#lang racket/base
;; Raw Stable C ABI bindings: library discovery, dlopen, and symbol
;; resolution. ffi/unsafe is quarantined in runtime/ffi/;
;; other modules go through the safe exports of this directory. Bindings
;; grow with the eager backend; today: version, device/dtype constants,
;; ctensor lifecycle (create / metadata / copy / item / delete), and the
;; symbol resolver. Lifecycle here is manual — explicit ctensor-delete!;
;; ops.rkt attaches finalizers to the ctensors it produces.

(require ffi/unsafe
         racket/runtime-path)

(provide ctensor?
         libtorch-available?
         libtorch-lib
         raw-fn
         lazy-fn
         check-code
         abi-version
         abi-version-string
         device-code
         dtype-code
         shape->strides
         make-ctensor
         bytes->ctensor
         ctensor->bytes
         ctensor-sizes
         ctensor-strides
         ctensor-dtype-code
         ctensor-device-code
         ctensor-storage-offset
         ctensor-data-ptr
         ctensor-copy!
         ctensor-item-f32
         ctensor-item-i64
         ctensor-delete!
         ctensor-reinterpret
         ctensor-clone
         ctensor-contiguous?
         real->ctensor)

;; A ctensor is an ATen tensor seen from C: an opaque pointer managed by
;; libtorch. This predicate is the identity the typed world uses for it
;; (tensor.rkt imports it as the opaque CTensor type).
(define ctensor? cpointer?)

(define-runtime-path this-dir ".")

;; Discovery: explicit override first, then the torch installed in the
;; project venv (.venv at the repo root) — the same installation that
;; serves reference generation and the future sidecar, so the Python side
;; and the dlopened dylibs cannot drift apart.
(define (venv-torch-lib-dir)
  (define venv-lib (simplify-path (build-path this-dir "../../.." ".venv/lib")))
  (and (directory-exists? venv-lib)
       (for/or ([d (in-list (directory-list venv-lib))])
         (define p (build-path venv-lib d "site-packages/torch/lib"))
         (and (regexp-match? #rx"^python3" (path->string d)) (directory-exists? p) p))))

(define lib-dir
  (let ([override (getenv "ZHULONG_LIBTORCH")])
    (if override
        (string->path override)
        (venv-torch-lib-dir))))

(define libtorch-path (and lib-dir (build-path lib-dir "libtorch_cpu.dylib")))

(define libtorch (and libtorch-path (file-exists? libtorch-path) (ffi-lib libtorch-path)))

(define (libtorch-available?)
  (and libtorch #t))

(define (libtorch-lib)
  (unless libtorch
    (error
     'libtorch-lib
     "libtorch not found (looked at ~a); run tools/setup-python.sh or set ZHULONG_LIBTORCH to a torch/lib directory"
     (or libtorch-path "<no venv detected>")))
  libtorch)

;; Resolve a C symbol against libtorch, e.g. (raw-fn "aoti_torch_mm_out" (_fun ...)).
(define (raw-fn name type)
  (get-ffi-obj name (libtorch-lib) type))

(define (raw-const name)
  (get-ffi-obj name (libtorch-lib) (_fun -> _int32) (lambda () #f)))

;; aoti_torch_abi_version packs major<<56 | minor<<48 (observed for 2.13.0).
(define (abi-version)
  ((raw-fn "aoti_torch_abi_version" (_fun -> _uint64))))

(define (abi-version-string)
  (define v (abi-version))
  (format "~a.~a.~a"
          (arithmetic-shift v -56)
          (bitwise-and (arithmetic-shift v -48) 255)
          (bitwise-and (arithmetic-shift v -40) 255)))

(define device-consts
  (hasheq 'cpu
          "aoti_torch_device_type_cpu"
          'mps
          "aoti_torch_device_type_mps"
          'cuda
          "aoti_torch_device_type_cuda"))

(define dtype-consts
  (hasheq 'f32
          "aoti_torch_dtype_float32"
          'f16
          "aoti_torch_dtype_float16"
          'bf16
          "aoti_torch_dtype_bfloat16"
          'i64
          "aoti_torch_dtype_int64"
          'i32
          "aoti_torch_dtype_int32"
          'u8
          "aoti_torch_dtype_uint8"
          'bool
          "aoti_torch_dtype_bool"))

(define (lookup-const who table key)
  (define name (hash-ref table key (lambda () (raise-arguments-error who "unknown key" "key" key))))
  (define f (raw-const name))
  (unless f
    (raise-arguments-error who "constant not exported by this libtorch" "symbol" name))
  (f))

;; Integer codes libtorch uses for devices and dtypes.
(define (device-code d)
  (lookup-const 'device-code device-consts d))
(define (dtype-code d)
  (lookup-const 'dtype-code dtype-consts d))

;; Symbols resolve on first call so the module loads without libtorch.
(define (lazy-fn name type)
  (define f #f)
  (lambda args
    (unless f
      (set! f (raw-fn name type)))
    (apply f args)))

(define (check-code who code)
  (unless (zero? code)
    (error who "libtorch returned error code ~a" code)))

(define c-empty-strided
  (lazy-fn "aoti_torch_empty_strided"
           (_fun _int64
                 (_list i _int64)
                 (_list i _int64)
                 _int32
                 _int32
                 _int32
                 (out : (_ptr o _pointer))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'make-ctensor code)
                   out))))

(define c-get-dim
  (lazy-fn "aoti_torch_get_dim"
           (_fun _pointer
                 (out : (_ptr o _int64))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'ctensor-sizes code)
                   out))))

(define c-get-sizes
  (lazy-fn "aoti_torch_get_sizes"
           (_fun _pointer
                 (out : (_ptr o _pointer))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'ctensor-sizes code)
                   out))))

(define c-get-strides
  (lazy-fn "aoti_torch_get_strides"
           (_fun _pointer
                 (out : (_ptr o _pointer))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'ctensor-strides code)
                   out))))

(define c-get-dtype
  (lazy-fn "aoti_torch_get_dtype"
           (_fun _pointer
                 (out : (_ptr o _int32))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'ctensor-dtype-code code)
                   out))))

(define c-get-device-type
  (lazy-fn "aoti_torch_get_device_type"
           (_fun _pointer
                 (out : (_ptr o _int32))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'ctensor-device-code code)
                   out))))

(define c-get-storage-offset
  (lazy-fn "aoti_torch_get_storage_offset"
           (_fun _pointer
                 (out : (_ptr o _int64))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'ctensor-storage-offset code)
                   out))))

(define c-get-data-ptr
  (lazy-fn "aoti_torch_get_data_ptr"
           (_fun _pointer
                 (out : (_ptr o _pointer))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'ctensor-data-ptr code)
                   out))))

(define c-copy
  (lazy-fn "aoti_torch_copy_"
           (_fun _pointer _pointer _int32 -> (code : _int32) -> (check-code 'ctensor-copy! code))))

(define c-item-f32
  (lazy-fn "aoti_torch_item_float32"
           (_fun _pointer
                 (out : (_ptr o _float))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'ctensor-item-f32 code)
                   out))))

(define c-item-i64
  (lazy-fn "aoti_torch_item_int64"
           (_fun _pointer
                 (out : (_ptr o _int64))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'ctensor-item-i64 code)
                   out))))

(define c-delete
  (lazy-fn "aoti_torch_delete_tensor_object"
           (_fun _pointer -> (code : _int32) -> (check-code 'ctensor-delete! code))))

;; Row-major strides for a given shape.
(define (shape->strides shape)
  (define-values (strides numel)
    (for/fold ([acc '()]
               [stride 1])
              ([d (in-list (reverse shape))])
      (values (cons stride acc) (* stride d))))
  strides)

(define (make-ctensor dtype shape #:strides [strides (shape->strides shape)])
  (c-empty-strided (length shape) shape strides (dtype-code dtype) (device-code 'cpu) 0))

;; Copying constructor: the bytes are memcpy'd in, so the tensor owns its
;; storage and the byte string may be collected. Zero-copy from-blob is
;; reserved for the mmap-backed loader, whose arena keeps the mapping alive.
(define (bytes->ctensor dtype shape bs)
  (define h (make-ctensor dtype shape))
  (memcpy (ctensor-data-ptr h) bs (bytes-length bs))
  h)

(define (ctensor->bytes h nbytes)
  (define out (make-bytes nbytes))
  (memcpy out (ctensor-data-ptr h) nbytes)
  out)

(define (ctensor-sizes h)
  (define n (c-get-dim h))
  (define p (c-get-sizes h))
  (for/list ([i (in-range n)])
    (ptr-ref p _int64 i)))

(define (ctensor-strides h)
  (define n (c-get-dim h))
  (define p (c-get-strides h))
  (for/list ([i (in-range n)])
    (ptr-ref p _int64 i)))

(define (ctensor-dtype-code h)
  (c-get-dtype h))
(define (ctensor-device-code h)
  (c-get-device-type h))
(define (ctensor-storage-offset h)
  (c-get-storage-offset h))
(define (ctensor-data-ptr h)
  (c-get-data-ptr h))

(define (ctensor-copy! dst src)
  (c-copy dst src 0))

(define (ctensor-item-f32 h)
  (c-item-f32 h))
(define (ctensor-item-i64 h)
  (c-item-i64 h))

(define (ctensor-delete! h)
  (c-delete h))

(define c-reinterpret
  (lazy-fn "aoti_torch__reinterpret_tensor"
           (_fun _pointer
                 _int64
                 (_list i _int64)
                 (_list i _int64)
                 _int64
                 (out : (_ptr o _pointer))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'ctensor-reinterpret code)
                   out))))

(define c-clone
  (lazy-fn "aoti_torch_clone"
           (_fun _pointer
                 (out : (_ptr o _pointer))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'ctensor-clone code)
                   out))))

(define c-is-contiguous
  (lazy-fn "aoti_torch_is_contiguous"
           (_fun _pointer
                 (out : (_ptr o _stdbool))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'ctensor-contiguous? code)
                   out))))

(define c-scalar-f64
  (lazy-fn "aoti_torch_scalar_to_tensor_float64"
           (_fun _double
                 (out : (_ptr o _pointer))
                 ->
                 (code : _int32)
                 ->
                 (begin
                   (check-code 'real->ctensor code)
                   out))))

;; A strided view sharing storage with h; both ctensors are deleted
;; independently (the storage is refcounted underneath).
(define (ctensor-reinterpret h sizes strides offset)
  (c-reinterpret h (length sizes) sizes strides offset))

(define (ctensor-clone h)
  (c-clone h))

(define (ctensor-contiguous? h)
  (c-is-contiguous h))

(define (real->ctensor x)
  (c-scalar-f64 (exact->inexact x)))
