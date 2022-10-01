#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require flomat)

;; ----------------------------------------------------

(require "activation.rkt")
(require "ga.rkt")
(require "layer.rkt")

;; ----------------------------------------------------

(provide model<%>

         ; model classes
         seq-model%

         ; macros
         define-model)

;; ----------------------------------------------------

(define model<%>
  (interface () predict))

;; ----------------------------------------------------

(define seq-model%
  (class* object% (heritable<%> model<%>)
    (super-new)

    ; constructor fields
    (init-field layers)

    ; return number of inputs and outputs
    (define/public (get-inputs) (send (first layers) get-inputs))
    (define/public (get-outputs) (send (last layers) get-outputs))

    ; send input sequence through all layers
    (define/public (predict X)
      (for/fold ([Z (matrix X)])
                ([layer layers])
        (send layer step Z)))

    ; create a new model crossing this with another
    (define/public (crossover model)
      (let ([layers-model (get-field layers model)])
        (new this% [layers (for/list ([a layers]
                                      [b layers-model])
                             (send a crossover b))])))))

;; ----------------------------------------------------

(define (build-layers layers inputs)
  (for/list ([ctor layers])
    (let ([layer (ctor inputs)])
      (begin0 layer (set! inputs (send layer get-outputs))))))

;; ----------------------------------------------------

(define-syntax (define-model stx)
  (syntax-case stx ()
    [(_ name class% [layer ...])
     #'(define (name inputs)
         (new class% [layers (build-layers (list layer ...) inputs)]))]
    [(_ name class% [layer ...] #:inputs inputs)
     #'(define (name)
         (new class% [layers (build-layers (list layer ...) inputs)]))]))

;; ----------------------------------------------------

(module+ test
  (require rackunit)

  ; create a new model builder
  (define-model dummy-model seq-model%
    [(dense-layer 2 .relu!)
     (dense-layer 3 .gaussian!)])

  ; ensure the shape of the model
  (let ([dummy (dummy-model 1)])
    (check-equal? (send dummy get-inputs) 1)
    (check-equal? (send dummy get-outputs) 3)

    ; ensure we can run it
    (void (send dummy predict '(0.5)))))
