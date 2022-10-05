#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require racket/generic)

;; ----------------------------------------------------

(require flomat)
(require plot)

;; ----------------------------------------------------

(require "activation.rkt")
(require "dnn.rkt")
(require "explore.rkt")
(require "ga.rkt")
(require "layer.rkt")
(require "loss.rkt")
(require "model.rkt")

;; ----------------------------------------------------

(provide (all-defined-out))

;; ----------------------------------------------------

(define dqn%
  (class dnn%
    (init model)

    ; constructor fields
    (init-field initial-state
                state->X
                do-action
                [gamma 0.9]
                [batch-size 500])

    ; create a curiosity (exploration) model
    (define explore (new explore% [model model]))

    ; random change at choosing a different action
    (define e-greedy (epsilon-greedy 0.1))

    ; batch count for this generation
    (define batch 0)

    ; define a custom agent class
    (define agent%
      (class* object% (recomb<%>)
        (super-new)

        ; constructor fields
        (init-field model

                    ; state fields
                    [state (initial-state)]
                    [reward 0]
                    [terminal? #f])

        ; revert the state of the agent
        (define/public (reset-state)
          (set! state (initial-state))
          (set! reward 0)
          (set! terminal? #f))

        ; run input vector
        (define/public (step [train? #f])
          (unless terminal?
            (let* ([X (state->X state)]
                   [Z (send model call X)]
                   [k (e-greedy Z)])
              (let-values ([(new-reward new-state new-terminal?) (do-action state k)])
                (set! state new-state)
                (set! terminal? new-terminal?)
                
                ; optionally reward and train
                (when train?
                  (let* ([Y (state->X new-state)]
                         [X+Y (stack X Y)]
                         [K (hot-encode k (size Z))]
                         [R (send explore get-curiosity X+Y K)])
                    (set! reward (+ reward (* new-reward (expt gamma batch))))))))))

        ; recombine with another agent
        (define/public (recomb other)
          (let ([model-other (get-field model other)])
            (new this% [model (send model recomb model-other)])))))

    ; initialize the population with agent models
    (super-new [model (λ () (new agent% [model (model)]))])

    ; allow use of the get-model method
    (inherit [get-agent get-model])
    
    ; allow use of all the models
    (inherit-field [agents models])

    ; return the state of the best agent
    (define/public (get-state)
      (get-field state (get-agent)))
    
    ; perform actions for each agent
    (define/public (train-agents #:watch? [watch? #f])
      (set! batch (add1 batch))

      ; true if all agents are terminal, batch is full, 
      (and (or (for/fold ([all-terminal? #t])
                         ([agent agents])
                 (and all-terminal? (begin
                                      (send agent step #t)
                                      (get-field terminal? agent))))
               
               ; batch full?
               (and batch-size (>= batch batch-size))
               
               ; watched agent in terminal state
               (and watch? (get-field terminal? (get-agent))))

           ; advance agents to the next generation
           (train)))

    ; advance to the next generation
    (define/override (train)
      (println 'and-here)
      (begin0 (next-gen! agents (λ (agent) (get-field reward agent)) >)

              ; reset batch
              (set! batch 0)

              ; reset agents
              (for ([agent agents])
                (send agent reset-state))))

    ; train agents for multiple generations
    (define/override (train+ #:generations [n 100])
      (let ([reward (for/list ([x n])
                      (do ([y (train-agents)
                              (train-agents)])
                        [y (list x y)]))])
        (plot (lines reward #:y-min 0) #:x-label "Generation" #:y-label "Reward")))))

;; ----------------------------------------------------

(module+ test
  (require racket/fixnum)
  
  ; This test is a simple game of "lights". There are 6 lights in an initial
  ; random state of either on or off. Each step, an agent can toggle the state
  ; of a given light (1-6). The goal is to turn all the lights on. Each agent
  ; only gets 6 turns.
  
  (struct state [turn lights] #:transparent)

  (define (initial-state)
    (state 1 (random #b1000000)))

  (define (bit n i)
    (fxand (fxrshift n i) 1))

  (define (state->X st)
    (matrix (for/list ([i 6]) (bit (state-lights st) i))))

  (define (popcount n)
    (for/sum ([i 6]) (if (zero? (bit n i)) 0 1)))

  (define (flip-switch st k)
    (match-let ([(state turn lights) st])
      (let* ([lights (fxxor lights (fxlshift 1 k))]
             [won? (= lights #b111111)])
        (values (bit lights k)                ; reward on turn on
                (state (add1 turn) lights)    ; updated state
                (or won? (= turn 6))))))      ; won or after last turn
  
  (define-seq-model lights-model
    [(dense-layer 10 .relu!)
     (dense-layer 6 .relu!)]
    #:inputs 6)

  (define dqn
    (new dqn%
         [model lights-model]
         [initial-state initial-state]
         [state->X state->X]
         [do-action flip-switch]))

  (send dqn train)
  #;(send dqn train+ #:generations 300)

  (define (play)
    (let ([agent (send dqn get-model)])
      (send agent reset-state)
      (let loop ()
        (send agent step)
        (match-let ([(state turn lights) (get-field state agent)])
          (displayln (~r lights #:base 2 #:min-width 6 #:pad-string "0")))
        (unless (get-field terminal? agent)
          (loop)))))
  )
