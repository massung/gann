#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require racket/generic)

;; ----------------------------------------------------

(require flomat)

;; ----------------------------------------------------

(require "activation.rkt")
(require "ga.rkt")
(require "layer.rkt")

;; ----------------------------------------------------

(provide (all-defined-out))

;; ----------------------------------------------------

(define model<%>
  (interface (layer<%>)))

;; ----------------------------------------------------

(define seq-model%
  (class* object% (model<%>)
    (super-new)

    ; constructor fields
    (init-field layers)

    ; return the shape of the model
    (define/public (get-inputs) (send (first layers) get-inputs))
    (define/public (get-outputs) (send (last layers) get-outputs))

    ; run input vector through all layers
    (define/public (call X)
      (for/fold ([Z X])
                ([layer layers])
        (send layer call Z)))

    ; recombine with another model
    (define/public (recomb other)
      (let ([layers-other (get-field layers other)])
        (new this% [layers (for/list ([a layers]
                                      [b layers-other])
                             (send a recomb b))])))))

;; ----------------------------------------------------

(define (seq-model . layers)
  (λ (inputs)
    (new seq-model%
         [layers (for/list ([layer layers])
                   (let ([layer (layer inputs)])
                     (begin0 layer (set! inputs (send layer get-outputs)))))])))

;; ----------------------------------------------------

(define-syntax (define-seq-model stx)
  (syntax-case stx ()
    [(_ name [layer-builder ...] #:inputs inputs)
     #'(define name
         (let ([builder (seq-model layer-builder ...)])
           (λ () (builder inputs))))]))

;; ----------------------------------------------------

(module+ test
  (require rackunit)

  ; create a new model builder
  (define-seq-model dummy-model
    [(dense-layer 2 .relu!)
     (dense-layer 3 .gaussian!)]
    #:inputs 1)

  ; test recombination of models
  (let ([a (dummy-model)]
        [b (dummy-model)])
    (void (send a recomb b)))

  ; ensure the shape of the model
  (let ([dummy (dummy-model)])
    (check-equal? (send dummy get-inputs) 1)
    (check-equal? (send dummy get-outputs) 3)

    ; ensure we can run it
    (void (send dummy call (column 0.5)))))
