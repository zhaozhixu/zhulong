#lang zhulong
;; SmolLM2 / Llama-family forward in branch-free dataflow form, faithful
;; to HF Llama semantics: rms-norm internals in f32, softmax in f32,
;; rope/mask tables built host-side in f32 and cast to the model dtype at
;; use. llama-forward returns every block output for layer parity.

(require (except-in racket/list argmax)
         racket/flonum
         zhulong/utils/safetensor)

(provide smollm2-135m
         llama-load
         llama-forward)

(define smollm2-135m
  (hasheq 'layers
          30
          'hidden
          576
          'heads
          9
          'kv-heads
          3
          'head-dim
          64
          'intermediate
          1536
          'vocab
          49152
          'rope-theta
          100000.0
          'eps
          1e-5))

;; --- host-side table building (f32, matching the reference chain) ---

(define (f32-tensor vals shape)
  (define bs (make-bytes (* 4 (length vals))))
  (for ([v (in-list vals)]
        [i (in-naturals)])
    (real->floating-point-bytes v 4 #f bs (* 4 i)))
  (bytes->tensor bs shape 'f32))

(define (rope-tables theta d max-len)
  (define half (quotient d 2))
  (define inv
    (for/vector ([j (in-range half)])
      (flsingle (/ 1.0 (flsingle (expt (exact->inexact theta) (exact->inexact (/ (* 2 j) d))))))))
  (define (table f)
    (for*/list ([p (in-range max-len)]
                [j (in-range d)])
      (flsingle (f (flsingle (* (exact->inexact p) (vector-ref inv (modulo j half))))))))
  (values (f32-tensor (table cos) (list max-len d)) (f32-tensor (table sin) (list max-len d))))

;; Additive causal mask; the masked value is bf16's most negative finite
;; number, the same constant the reference uses.
(define bf16-min -3.3895313892515355e38)

(define (causal-mask max-len)
  (f32-tensor (for*/list ([i (in-range max-len)]
                          [j (in-range max-len)])
                (if (> j i) bf16-min 0.0))
              (list max-len max-len)))

(define (ids->tensor ids)
  (define bs (make-bytes (* 8 (length ids))))
  (for ([t (in-list ids)]
        [i (in-naturals)])
    (integer->integer-bytes t 8 #t #f bs (* 8 i)))
  (bytes->tensor bs (list 1 (length ids)) 'i64))

;; --- weights ---

(define (llama-load path cfg #:max-len [max-len 64])
  (define st (safetensor-open path))
  (define (w name)
    (define key (string->symbol name))
    (bytes->tensor (safetensor-bytes st key) (safetensor-shape st key) (safetensor-dtype st key)))
  (define layers
    (for/vector ([i (in-range (hash-ref cfg 'layers))])
      (define (lw s)
        (w (format "model.layers.~a.~a.weight" i s)))
      (hasheq 'ln1
              (lw "input_layernorm")
              'q
              (lw "self_attn.q_proj")
              'k
              (lw "self_attn.k_proj")
              'v
              (lw "self_attn.v_proj")
              'o
              (lw "self_attn.o_proj")
              'ln2
              (lw "post_attention_layernorm")
              'gate
              (lw "mlp.gate_proj")
              'up
              (lw "mlp.up_proj")
              'down
              (lw "mlp.down_proj"))))
  (define-values (cos sin) (rope-tables (hash-ref cfg 'rope-theta) (hash-ref cfg 'head-dim) max-len))
  (hasheq 'cfg
          cfg
          'embed
          (w "model.embed_tokens.weight")
          'norm
          (w "model.norm.weight")
          'layers
          layers
          'cos
          cos
          'sin
          sin
          'mask
          (causal-mask max-len)))

;; --- layers ---

(define-layer (llama-rms-norm x w eps)
  (define x32 (to-dtype x #:dtype 'f32))
  (define n (mul x32 (rsqrt (add (mean (sqr x32) #:axis -1 #:keep-dim? #t) eps))))
  (mul w (to-dtype n #:dtype 'bf16)))

(define-layer (linear x w) (matmul x (transpose w #:dim0 0 #:dim1 1)))

(define-layer (rotate-half x half)
  (cat (neg (narrow x #:axis -1 #:start half #:len half))
       (narrow x #:axis -1 #:start 0 #:len half)
       #:axis -1))

(define-layer (rope-rotate x cos sin half) (add (mul x cos) (mul (rotate-half x half) sin)))

(define-layer (llama-attention x lw cos sin mask cfg)
  (define heads (hash-ref cfg 'heads))
  (define kv-heads (hash-ref cfg 'kv-heads))
  (define hd (hash-ref cfg 'head-dim))
  (define half (quotient hd 2))
  (define L (second (tensor-shape x)))
  (define (split t n)
    (transpose (reshape t #:shape (list 1 L n hd)) #:dim0 1 #:dim1 2))
  (define (expand t)
    (reshape (broadcast-to (unsqueeze t #:axis 2)
                           #:shape (list 1 kv-heads (quotient heads kv-heads) L hd))
             #:shape (list 1 heads L hd)))
  (define q (rope-rotate (split (linear x (hash-ref lw 'q)) heads) cos sin half))
  (define k (rope-rotate (split (linear x (hash-ref lw 'k)) kv-heads) cos sin half))
  (define v (expand (split (linear x (hash-ref lw 'v)) kv-heads)))
  (define scores
    (mul (matmul q (transpose (expand k) #:dim0 -2 #:dim1 -1)) (/ 1.0 (sqrt (exact->inexact hd)))))
  (define probs
    (to-dtype (softmax (to-dtype (add scores mask) #:dtype 'f32) #:axis -1) #:dtype 'bf16))
  (define ctx (transpose (matmul probs v) #:dim0 1 #:dim1 2))
  (linear (reshape ctx #:shape (list 1 L (* heads hd))) (hash-ref lw 'o)))

(define-layer (llama-mlp x lw)
  (linear (mul (silu (linear x (hash-ref lw 'gate))) (linear x (hash-ref lw 'up)))
          (hash-ref lw 'down)))

(define-layer (llama-block x lw cos sin mask cfg eps)
  (define h (add x (llama-attention (llama-rms-norm x (hash-ref lw 'ln1) eps) lw cos sin mask cfg)))
  (add h (llama-mlp (llama-rms-norm h (hash-ref lw 'ln2) eps) lw)))

;; --- forward ---

;; Returns (values hidden-states logits): hidden-states is the embedding
;; output followed by every block output, for layer-by-layer parity.
(define (llama-forward m ids)
  (define cfg (hash-ref m 'cfg))
  (define eps (hash-ref cfg 'eps))
  (define L (length ids))
  (define (clip t)
    (narrow t #:axis 0 #:start 0 #:len L))
  (define cos (to-dtype (clip (hash-ref m 'cos)) #:dtype 'bf16))
  (define sin (to-dtype (clip (hash-ref m 'sin)) #:dtype 'bf16))
  (define mask (to-dtype (narrow (clip (hash-ref m 'mask)) #:axis 1 #:start 0 #:len L) #:dtype 'bf16))
  (define h0 (embedding (hash-ref m 'embed) (ids->tensor ids)))
  (define hs
    (for/fold ([acc (list h0)]) ([lw (in-vector (hash-ref m 'layers))])
      (cons (llama-block (first acc) lw cos sin mask cfg eps) acc)))
  (define hn (llama-rms-norm (first hs) (hash-ref m 'norm) eps))
  (values (reverse hs) (linear hn (hash-ref m 'embed))))
