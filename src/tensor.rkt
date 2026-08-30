#lang typed/racket

(require "shape.rkt")

(provide Shape
         shape?
         shape-numel
         broadcast-shapes
         Dtype
         dtype?
         dtype-size
         Device
         device?
         current-device
         Tensor
         tensor?
         tensor-shape
         tensor-dtype
         tensor-rank
         tensor-numel
         (struct-out real-tensor)
         (struct-out sym-tensor))

(define-type Dtype (U 'f32 'f16 'bf16 'i64 'i32 'u8 'bool))

(define-predicate dtype? Dtype)

(: dtype-size (-> Dtype Positive-Integer))
(define (dtype-size d)
  (hash-ref dtype-sizes d))

(: dtype-sizes (Immutable-HashTable Dtype Positive-Integer))
(define dtype-sizes (hasheq 'f32 4 'f16 2 'bf16 2 'i64 8 'i32 4 'u8 1 'bool 1))

(define-type Device (U 'cpu 'mps 'cuda))

(define-predicate device? Device)

;; Default placement for newly created tensors only; never consulted for
;; dispatch or placement (operators follow their inputs; multi-device
;; placement is a phase-4 plan annotation on the graph).
(: current-device (Parameterof Device))
(define current-device (make-parameter 'cpu))

;; The parent struct carries the shared fields, so tensor-shape /
;; tensor-dtype / tensor? are plain struct accessors — no dispatch. The
;; parent constructor is not exported: a tensor is always a real-tensor or
;; a sym-tensor, and Tensor stays the closed union so matches can be
;; exhaustive.
(struct tensor
  ([shape : Shape] [dtype : Dtype])
  #:transparent)
(struct real-tensor
  tensor
  ([handle : Any] [device : Device])
  #:transparent)
(struct sym-tensor
  tensor
  ([id : Natural])
  #:transparent)

(define-type Tensor (U real-tensor sym-tensor))

(: tensor-rank (-> Tensor Index))
(define (tensor-rank t)
  (length (tensor-shape t)))

(: tensor-numel (-> Tensor Nonnegative-Integer))
(define (tensor-numel t)
  (shape-numel (tensor-shape t)))
