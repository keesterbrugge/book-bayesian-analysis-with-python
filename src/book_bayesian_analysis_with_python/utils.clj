(ns book-bayesian-analysis-with-python.utils
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py :refer [py.]]
            [tech.ml.dataset :as ds]))

;; try out plot functionality from gigasquid blo
;; https://gigasquidsoftware.com/blog/2020/01/18/parens-for-pyplot/

(require '[clojure.java.shell :as sh])

;;; This uses the headless version of matplotlib to generate a graph then copy it to the JVM
;; where we can then print it

;;;; have to set the headless mode before requiring pyplot
(def mplt (py/import-module "matplotlib"))
(py. mplt "use" "Agg")

(require-python 'matplotlib.pyplot)
(require-python 'matplotlib.backends.backend_agg)


(defmacro with-show
  "Takes forms with mathplotlib.pyplot to then show locally"
  [& body]
  `(let [_# (matplotlib.pyplot/clf)
         fig# (matplotlib.pyplot/figure)
         agg-canvas# (matplotlib.backends.backend_agg/FigureCanvasAgg fig#)]
     ~(cons 'do body)
     (py. agg-canvas# "draw")
     (matplotlib.pyplot/savefig "temp.png")
     (sh/sh "open" "temp.png")))

(comment

  (py/import-as numpy np)
  (let [x (np/arange 0 (* 3 np/pi) 0.1)
        y (np/sin x)]
    (with-show
      (matplotlib.pyplot/plot x y))) ;; NOTE works :)



  )

#_(defmacro quick-trace2 ;; TODO for some reason importing from here doesn't work. 
  [& body]
  
  (let [
        bindings (partition 2 body)   
        bindings' (mapcat (fn [[symb ls]]
                            [symb (concat (list(first ls) (name symb)) (rest ls))]) bindings)]
    (println bindings')
    `(py/with
      [_# (pm/Model)]
      (let [~@bindings']
        (pm/sample 1000)))))


(defn group-by-columns-and-aggregate [gr-colls agg-fns-map ds]
  (->> (ds/group-by identity ds gr-colls)
       (map (fn [[group-idx group-ds]]
              (into group-idx (map (fn [[k agg-fn]] [k (agg-fn group-ds)]) agg-fns-map))))
       ds/->dataset))


(defn plot-posterior-predictive-check [{:keys [trace model]} {:keys [xlim]}]
  (let [prior (pm/sample_prior_predictive :model model)
        posterior-pred (pm/sample_posterior_predictive :trace trace
                                                       :model model :samples 100)
        az-inf-obj (az/from_pymc3 :trace trace
                                           :prior prior
                                           :posterior_predictive posterior-pred)]
    (with-show
      (az/plot_ppc az-inf-obj)
      (when xlim
        (apply matplotlib.pyplot/xlim xlim)))))
