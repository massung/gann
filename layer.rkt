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
(require "loss.rkt")

;; ----------------------------------------------------

(provide layer<%>

         ; layer classes
         dense-layer%
         recurrent-layer%
         residual-layer%

         ; layer constructors
         dense-layer
         recurrent-layer
         residual-layer)

;; ----------------------------------------------------

(define layer<%>
  (interface (heritable<%>) step))

;; ----------------------------------------------------

(define dense-layer%
  (class* object% (layer<%>)
    (super-new)

    ; constructor fields
    (init-field weights
                bias
                activation)

    ; return number of inputs and outputs
    (define/public (get-inputs) (ncols weights))
    (define/public (get-outputs) (nrows weights))
    
    ; run input vector
    (define/public (step X)
      (activation (plus! (times weights X) bias)))

    ; create a new layer crossing weights and biases
    (define/public (crossover other)
      (let* ([bias-other (get-field bias other)]
             [weights-other (get-field weights other)])
        (new this%
             [activation activation]
             [bias (.mutate! (recomb bias bias-other))]
             [weights (.mutate! (recomb weights weights-other))])))))

;; ----------------------------------------------------

(define recurrent-layer%
  (class dense-layer%
    (super-new)

    ; access superclass fields
    (inherit-field weights bias activation)

    ; previous outputs
    (define v (make-vector (nrows weights)))

    ; memory is attached to the output
    (define/override (get-outputs) (* (nrows weights) 2))

    ; run input vector
    (define/override (step X)
      (let ([z (flomat->vector (activation (plus! (times weights X) bias)))])
        (begin0 (matrix (vector-append z v))

                ; save the output to memory
                (set! v z))))))

;; ----------------------------------------------------

(define residual-layer%
  (class dense-layer%
    (super-new)

    ; access superclass fields
    (inherit-field weights bias activation)

    ; outputs of a residual layer must match the number of inputs
    (unless (= (ncols weights) (nrows weights))
      (error "Residual layer outputs must match inputs!"))
    
    ; run input vector
    (define/override (step X)
      (let ([Z (plus! (times weights X) bias)])
        (activation (.+! Z X))))))

;; ----------------------------------------------------

(define (dense-layer outputs [activation .relu!])
  (λ (inputs)
    (let ([b (.-! (rand outputs 1) (rand outputs 1))]
          [w (.-! (rand outputs inputs) (rand outputs inputs))])
      (new dense-layer%
           [weights w]
           [bias b]
           [activation activation]))))

;; ----------------------------------------------------

(define (residual-layer [activation .relu!])
  (λ (inputs)
    (let ([b (.-! (rand inputs 1) (rand inputs 1))]
          [w (.-! (rand inputs) (rand inputs))])
      (new residual-layer%
           [weights w]
           [bias b]
           [activation activation]))))

;; ----------------------------------------------------

(define (recurrent-layer outputs [activation .relu!])
  (λ (inputs)
    (let ([b (.-! (rand outputs 1) (rand outputs 1))]
          [w (.-! (rand outputs inputs) (rand outputs inputs))])
      (new recurrent-layer%
           [weights w]
           [bias b]
           [activation activation]))))
