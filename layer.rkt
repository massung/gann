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

;; ----------------------------------------------------

(provide call<%>
         layer<%>

         ; layer classes
         dense-layer%
         recurrent-layer%

         ; layer builders
         dense-layer
         recurrent-layer)

;; ----------------------------------------------------

(define call<%>
  (interface () call))

;; ----------------------------------------------------

(define layer<%>
  (interface (call<%> recomb<%>) get-inputs get-outputs))

;; ----------------------------------------------------

(define dense-layer%
  (class* object% (layer<%>)
    (super-new)

    ; constructor fields
    (init-field weights
                bias
                activation)

    ; return the number of inputs and outputs
    (define/public (get-inputs) (ncols weights))
    (define/public (get-outputs) (nrows weights))

    ; process input vector
    (define/public (call X)
      (activation (plus! (times weights X) bias)))

    ; recombine with another layer
    (define/public (recomb other)
      (new this%
           [activation activation]
           [bias (.mutate! (crossover bias (get-field bias other)))]
           [weights (.mutate! (crossover weights (get-field weights other)))]))))

;; ----------------------------------------------------

(define recurrent-layer%
  (class dense-layer%
    (super-new)
    
    ; access superclass fields
    (inherit-field weights bias activation)

    ; previous outputs
    (define v (zeros (nrows weights) 1))

    ; attack previous outputs
    (define/override (get-outputs) (* (nrows weights) 2))

    ; process input vector
    (define/override (call X)
      (let ([z (super call X)])
        (begin0 (stack z v)

                ; save output to memory
                (set! v z))))))

;; ----------------------------------------------------

(define (<layer> class% outputs [activation .relu!])
  (λ (inputs)
    (let ([b (.-! (rand outputs 1) (rand outputs 1))]
          [w (.-! (rand outputs inputs) (rand outputs inputs))])
      (new class%
           [weights w]
           [bias b]
           [activation activation]))))

;; ----------------------------------------------------

(define dense-layer (curry <layer> dense-layer%))
(define recurrent-layer (curry <layer> recurrent-layer%))
