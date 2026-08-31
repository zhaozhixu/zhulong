#lang racket/base
;; DSL syntax layer v0: define-layer / define-model expand to a plain define
;; plus a definition-info record (name, kind, source location — the embryo of
;; provenance). with-eager and boundary weaving arrive with the tracer.

(require (for-syntax racket/base
                     syntax/parse)
         racket/contract
         syntax/location)

(provide define-layer
         define-model
         (contract-out (struct def-info ([kind symbol?] [srcloc srcloc?]))
                       [definition-info (-> symbol? (or/c def-info? #f))]
                       [definition-names (-> (listof symbol?))]))

(struct def-info
  (kind srcloc)
  #:transparent)

(define definitions (make-hasheq))

(define (definition-register! name kind loc)
  (hash-set! definitions name (def-info kind loc)))

(define (definition-info name)
  (hash-ref definitions name #f))
(define (definition-names)
  (sort (hash-keys definitions) symbol<?))

(define-syntax (define-layer stx)
  (syntax-parse stx
    [(_ (name:id . formals) body ...+)
     #'(begin
         (define (name . formals)
           body ...)
         (definition-register! 'name 'layer (quote-srcloc name)))]))

(define-syntax (define-model stx)
  (syntax-parse stx
    [(_ (name:id . formals) body ...+)
     #'(begin
         (define (name . formals)
           body ...)
         (definition-register! 'name 'model (quote-srcloc name)))]))
