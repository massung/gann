#lang racket

#|

Genetic Neural Networks for Racket.

Copyright (c) 2022 by Jeffrey Massung
All rights reserved.

|#

(require racket/random)

;; gann library
(require "../main.rkt")

#|

A pattern of N random colors will be chosen and becomes the secret
code. Each turn the DQN agent will guess N colors. As there are 5
colors to choose from, the action space is the 5^N.

The input is the known state of the board: all guesses and answer
keys. Guesses and keys are hot-encoded, meaning that if the colors
are '(R G B Y W), R will be encoded as '(1 0 0 0 0) and G will be
encoded as '(0 1 0 0 0). Possible key values are '(X ! ?) where X
not found in answer and ! is at the correct location and ? is wrong
location.

There are a maximum of T turns (guesses) that can be made, so the
input vector space is (5N+3N)*T+K, where K is the maximum number of
guesses allowed to be taken.

The rewards for Mastermind are sparse and a large part of the AI
learning is about trying to learn new things about the state: making
the same guess over and over reveals no new information about what
the solution to the puzzle is.

|#

(require flomat)

;; number of colors in the answer
(define N 2)

;; number of guesses allowed
(define max-guesses 10)

;; valid colors
(define colors '(R G B))

;; space for a color, guess, key, turn, and game
(define color-space (length colors))
(define guess-space (* color-space N))
(define key-space (* N 3))
(define turn-space (+ guess-space key-space))
(define input-space (+ (* turn-space max-guesses) max-guesses))

;; all possible actions
(define action-space
  (list->vector (apply cartesian-product (make-list N colors))))

;; number of possible actions
(define action-count
  (vector-length action-space))

;; hot encode a value from a list of options
(define (hot-encode value xs)
  (for/list ([x xs])
    (if (eq? x value) 1 0)))

;; state of the game
(struct state [ans guesses keys input turn] #:transparent)

;; create a new state
(define (new-state [ans (random-sample colors N)])
  (state ans '() '() (zeros input-space 1) 0))

;; convert state to inputs
(define (state->X st)
  (begin0 (state-input st)

          ; reset the turn input vector
          (for ([i max-guesses])
            (mset! (state-input st) i 0 (if (= i (state-turn st)) 1 0)))))

;; given an answer and guess, return the keys
(define (key-for-guess ans guess)
  (let* ([!s (for/list ([g guess] [a ans] #:when (eq? g a)) g)]
         [Xs (for/list ([g guess] #:unless (memq g ans)) g)]
         [?s (for/list ([g guess] #:unless (or (memq g !s) (memq g Xs))) g)])
    (append (map (const '!) !s)
            (map (const '?) ?s)
            (map (const 'X) Xs))))

;; update an input vector with hot-encoded values for a turn
(define (update-input! v turn guess key)
  (let ([turn-offset (* turn-space turn)])
    (for ([(g i) (in-indexed guess)])
      (mset! v (+ turn-offset (* i color-space) (index-of colors g)) 0 1))
    (for ([(k i) (in-indexed key)])
      (mset! v (+ turn-offset guess-space (* i 3) (index-of '(X ! ?) k)) 0 1))))

;; make a guess and return the reward, state, and terminal flag
(define (make-guess st guess)
  (match-let ([(state ans guesses keys input turn) st])
    (let* ([key (key-for-guess ans guess)]

           ; create the new state
           [nst (state ans (cons guess guesses) (cons key keys) (copy-flomat input) (add1 turn))]

           ; is it game over?
           [won? (equal? guess ans)]
           [lost? (>= (state-turn nst) max-guesses)]

           ; was a duplicate guess made? penalize those
           [dup? (member guess guesses)]

           ; reward for this state
           [reward (if won? 100 0)])
#|
      [lost? 0]         ; failure reward (sparse)
                     [won? 100]        ; victory reward (sparse)
                     [dup? 0]          ; no new information, that's bad
                     [else             ; key reward (dense)
                      (if (empty? keys)
                          #f
                          (for/fold ([n #f])
                                    ([k key] [p (first keys)] #:unless (eq? k p))
                            (match (cons p k)
                              [(cons _ 'X) (+ (or n 0) 1)]          ; gaining X is probably bad
                              [(cons _ '!) (+ (or n 0) 5)]          ; gaining ! is good
                              [(cons _ '?) (+ (or n 0) 2)])))])])   ; gaining ? can be good
  |#    
      ; update the input vector for the state
      (update-input! (state-input nst) turn guess key)

      ; return the reward, new state, and terminal flag
      (values reward nst (or won? lost?)))))

;; choose the guess from the action space and do it
(define (perform-action st k)
  (make-guess st (vector-ref action-space k)))

;; how many nodes in the hidden layer(s)
(define hidden-layer-n
  (floor (* (+ input-space action-count max-guesses) 3/4)))

;; create a simple xor model
(define-seq-model mastermind-model
  [(<dense-layer> hidden-layer-n .relu!)
   ;(dense-layer hidden-layer-n .relu!)
   (<dense-layer> action-count .relu!)]
  #:inputs input-space)

;; create the DQN to train
(define dqn (new dqn%
                 [model mastermind-model]
                 [population-size 100]
                 [initial-state new-state]
                 [state->X state-input]
                 [do-action perform-action]))

;; 
(define (train [n 100])
  (send dqn train+ #:generations n))

;; pretty-print the output of a state
(define (print-state st)
  (match-let ([(state ans guesses keys input turn) st])
    (for ([guess (reverse guesses)]
          [key (reverse keys)])
      (displayln (format "~a -> ~a" guess key)))
    (displayln "------------------")
    (displayln ans)))

;; run through a single game
(define (play)
  (let take-turn ()
    (unless (send dqn predict)
      (take-turn)))
  (print-state (send dqn get-state)))

;; train, play, train, play, etc...
(define (train-and-play [n 100])
  (for ([i n])
    (train)
    (play)))
