#lang zhulong
;; Shared layer library (application level). v0: rms-norm — the design
;; document's §3.1 example, exercising the full #lang zhulong chain.

(provide rms-norm)

(define-layer (rms-norm x #:eps [eps 1e-6])
  (mul x (rsqrt (add (mean (sqr x) #:axis -1 #:keep-dim? #t) eps))))
