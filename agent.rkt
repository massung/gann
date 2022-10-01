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
(require "model.rkt")

;; ----------------------------------------------------

(provide (all-defined-out))

;; ----------------------------------------------------

(define agent%
  (class* object% (heritable<%> model<%>)
    (super-new)

    ; constructor fields
    [init-field model initial-state]

    ; maintain rewards, current state, and terminal flag
    (field [rewards '()]
           [state (initial-state)]
           [terminal? #f])

    ; reset the agent
    (define/public (reset-state)
      (set! rewards '())
      (set! state (initial-state))
      (set! terminal? #f))

    ; choose an action, perform it, and update state
    (define/public (predict proc state->X #:explore [explore greedy] #:train? [train? #f])
      (unless terminal?
        (let* ([X (state->X state)]
               [Z (send model predict X)]
               [k (explore (flomat->vector Z))])
          (let-values ([(reward new-state terminal-state?) (proc state k)])
            (set! state new-state)
            (set! terminal? terminal-state?)
            
            ; track rewards
            (when (and train? reward)
              (set! rewards (cons reward rewards))))))

      ; true if in a terminal state
      terminal?)

    ; create a new agent with a new model
    (define/public (crossover agent)
      (let ([new-model (send model crossover (get-field model agent))])
        (new this%
             [model new-model]
             [initial-state initial-state])))))

;; ----------------------------------------------------

(define (agent-fitness gamma)
  (λ (agent)
    (let* ([rewards (get-field rewards agent)]
           [n (length rewards)]
           [dev (/ 2.0 (sqrt (* 2.0 pi)))])
      
      ; sum the rewards using a gaussian weight
      (for/sum ([(x i) (in-indexed rewards)])
        (let ([k (/ (- (/ i n) (- 1.0 gamma)) 0.5)])
          (* x dev (exp (- (* k k)))))))))

;; ----------------------------------------------------

(define (greedy Z)
  (for/fold ([k #f]
             [q #f] #:result k)
            ([(z i) (in-indexed Z)])
    (if (or (not q) (> z q))
        (values i z)
        (values k q))))

;; ----------------------------------------------------

(define (epsilon-greedy [epsilon 0.05] [n #f])
  (let ([e-greedy (λ (x Z)
                    (if (< (random) (* epsilon x))

                        ; just choose a random index from the output vector
                        (random (vector-length Z))
                
                        ; fallback to greedy
                        (greedy Z)))])
    (if n

        ; declay epsilon with gaussian curve
        (let ([g 0])
          (λ (Z)
            (begin0 (e-greedy (exp (- (* (/ g n 0.5) (/ g n 0.5)))) Z)

                    ; decay exploration with gaussian curve
                    (set! g (min (add1 g) n)))))

        ; constant epsilon
        (curry e-greedy 1.0))))

;; ----------------------------------------------------

(define (boltzmann Z)
  (let ([n (random)])
    (for/fold ([k #f]
               [q 0.0] #:result k)
              ([(z i) (in-indexed Z)] #:break (< n q))
      (values i (+ q z)))))

;; ----------------------------------------------------

(module+ test
  (require rackunit)
  (require plot)

  ; ensure greedy works
  (check-equal? (greedy '(0 1 2 3)) 3)
  (check-equal? (greedy '(3 2 1 0)) 0)

  ; ensure boltzmann has the proper distribution
  (parameterize ([current-pseudo-random-generator (make-pseudo-random-generator)])
    (random-seed 0)

    ; create a histogram of exepected probabilities
    (let* ([n 1000]
           [Z #(0.2 0.4 0.3 0.1)]
           [R (vector-map (const 0) Z)])
      (for ([_ n])
        (let ([i (boltzmann Z)])
          (vector-set! R i (add1 (vector-ref R i)))))

    ; plot how often each index was selected
    (plot (discrete-histogram (for/list ([x Z] [y R]) (list x y)) #:y-max 1000)))))
