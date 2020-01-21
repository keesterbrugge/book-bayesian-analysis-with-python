(ns book-bayesian-analysis-with-python.1
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py :refer [py. py.- py..]]
            [oz.core :as oz]))

;; (py/finalize!)
;; (py/initialize!)
(py/import-as pymc3 pm)
;; (require-python '([pymc3 :as pm]))
(require-python '([builtins :as pyb]))

(py/$. pm "__version__") 
(py.- pm "__version__")
;; => "3.8"

(pyb/help pm)

;; first example and ways of runnig it
(-> (py/run-simple-string "import scipy; x = scipy.stats.norm(0,1).rvs(3)")
    :globals
    (get "x"))

(require-python 'scipy.stats)
(py/$a (scipy.stats/norm 0 1) rvs 3)

(require-python '[numpy :as np])

(defn fig1-1-spec [ds]
  {:data {:values ds}
   :mark :line
   :encoding {:x {:field :x}
              :y {:field :pdf}
              :row {:field :mu}
              :column {:field :sd}}
   :resolve {:axis {:x :independent :y :independent}}})

(def fig1-1-ds
  (mapcat seq (for [mu [-1 0 1]
                sd [0.5 1 1.5]
                ]
            (let [x (np/linspace -7 7 200)
                  pdf (-> (scipy.stats/norm :loc mu :scale sd)
                          (py/$a pdf x ))]
              (map #(hash-map :x %1 :pdf %2 :mu mu :sd sd) x pdf)
              ))))

(oz/view! (fig1-1-spec fig1-1-ds))



(def fig1-3-ds
  (let [n-params [1 2 4]
        p-params [0.25 0.5 0.75]
        xs (np/arange 0 (inc (np/max n-params)))]
    (for [n n-params p p-params x xs]
      ;; {:pmf (py/$a (scipy.stats/binom n p) pmf x)
      {:pmf (py. (scipy.stats/binom n p) pmf x)
       :x x
       :n n
       :p p})))

(defn fig1-3-spec [ds]
  {:data {:values ds}
   :encoding {:x {:field :x}
              :y {:field :pmf}
              :row {:field :n}
              :column {:field :p}}
   :mark :bar
   :resolve {:axis {:x :independent :y :independent}}})

(oz/view! (fig1-3-spec fig1-3-ds))

(require 'clojure.java.io)

(clojure.java.io/file "/Users/keesterbrugge/Downloads/clojure.png" )
