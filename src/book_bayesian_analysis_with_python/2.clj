(ns book-bayesian-analysis-with-python.2
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py :refer [py. py.- py..]]
            [book-bayesian-analysis-with-python.utils :refer [with-show]]
            [oz.core :as oz]
            [tech.ml.dataset :as ds]
            [tech.ml.dataset.pipeline :as ds-pipe]
                     ))

(require-python '[builtins :as pyb]
                '[pymc3 :as pm :bind-ns]
                '[numpy :as np]
                '[arviz :as az]
                '[scipy.stats :as stats :refer [bernoulli]]
                'operator)


(pyb/help pm)
(assert (= 11 (np/dot [1 2 ] [ 3 4])))
(py.- pm "__version__")
(py. bernoulli rvs :p 0.4 :size 5)











(require-python '([scipy.stats :as stats :refer [bernoulli]]))
(py. np/random seed 123)
(def trials 4)
(def theta-real 0.35)
(def data (py. bernoulli rvs :p theta-real :size trials))

(def trace (py/with
            [our-first-model (py. pm Model)]
            (let [theta (py. pm Beta "theta" :alpha 1 :beta 1)
                  y (py. pm Bernoulli "y" :p theta :observed data)
                  trace (py. pm sample 1000 :random_seed 123)]
              trace)))


(az/summary trace)

(with-show (az/plot_trace trace))






(def data-chemical-shifts (np/loadtxt "./resources/data/chemical_shifts.csv"))

(let [ds (map #(zipmap [:x] %&) data-chemical-shifts)
      viz-spec {:data {:values ds}
                :mark :area
                :transform [{:density :x :bandwith 0.3}]
                :encoding {:x {:field :value :type :quantitative}
                           :y {:field :density :type :quantitative}}}
      ]
  (oz/view! viz-spec))

(def trace-g (py/with
            [model-g (py. pm Model)]
            (let [
                  mu (py. pm Uniform "mu" :lower 40 :upper 70)
                  sigma (py. pm HalfNormal "sigma" :sd 10)
                  y (py. pm Normal "y" :mu mu :sd sigma :observed data-chemical-shifts)
                  trace (py. pm sample 1000 :random_seed 123)]
              trace)))

(py/->jvm (az/summary trace-g))

(with-show (az/plot_trace trace-g))

(with-show (az/plot_joint trace-g :kind :kde :fill_last false))

(py/get-item trace-g :sigma)

;; (:sigma trace-g)  ;; doensn't work. 







(def inference-g (py/with
              [model-g (py. pm Model)]
              (let [
                    mu (py. pm Uniform "mu" :lower 40 :upper 70)
                    sigma (py. pm HalfNormal "sigma" :sd 10)
                    y (py. pm Normal "y" :mu mu :sd sigma :observed data-chemical-shifts)
                    trace (py. pm sample 1000 :random_seed 123)]
                {:trace  trace
                 :model model-g})))


(def y-pred-g
  (pm/sample_posterior_predictive (:trace inference-g) 100 (:model inference-g)))

(let [data-ppc (az/from_pymc3 :trace (:trace inference-g)
                              :posterior_predictive y-pred-g)]
  (with-show (az/plot_ppc data-ppc)))



(def inference-g (py/with
                  [model-g (py. pm Model)]
                  (let [
                        mu (py. pm Uniform "mu" :lower 40 :upper 70)
                        sigma (py. pm HalfNormal "sigma" :sd 10)
                        y (py. pm Normal "y" :mu mu :sd sigma :observed data-chemical-shifts)
                        trace (py. pm sample 1000 :random_seed 123)]
                    {:trace  trace
                     :model model-g})))


(py. (py. scipy.stats t :loc 0 :scale 1 :df 1) rvs 100)


(require-python '([scipy.stats :as stats :refer [t bernoulli]]))
(py. t             rvs :df 1 :size 3)
(py. stats/t       rvs :df 1 :size 3)
(py. scipy.stats/t rvs :df 1 :size 3)

(np/mean (py. t rvs :df 0.9 :size 100))



(def inference-gt (py/with
                  [model-gt (py. pm Model)]
                  (let [
                        mu (py. pm Uniform "mu" :lower 40 :upper 75)
                        sigma (py. pm HalfNormal "sigma" :sd 10)
                        nu (py. pm Exponential "nu" :lam 1/30 )
                        y (py. pm StudentT "y" :mu mu :sd sigma :nu nu
                               :observed data-chemical-shifts)
                        trace (py. pm sample 1000 :random_seed 123)]
                    {:trace  trace
                     :model model-gt})))

(with-show (az/plot_trace (:trace inference-gt)))

(let [{:keys [trace model]} inference-gt
      y-ppc-t (pm/sample_posterior_predictive trace 100 model :random_seed 123)
      y-pred-t (az/from_pymc3 :trace trace :posterior_predictive y-ppc-t)]
  (with-show
    (az/plot_ppc y-pred-t)
    (matplotlib.pyplot/xlim 40 70)))


(def tips (ds/->dataset "resources/data/tips.csv"))

(take 5 tips) ;; get first  5 columns

(ds/select tips :all (range 5)) ;; first 5 rows

(let [ds (ds/->flyweight tips)
      spec {:data  {:values ds}
            :mark :area
            :transform [{:density :tip :groupby [:day]}]
            :encoding {:x {:field :value :type :quantitative}
                       :y {:field :density :type :quantitative}
                       :color {:field :day :type :nominal}
                       :row {:field :day :type :nominal}}}]
  (oz/view! spec))







(require-python 'pandas)

(def tips-pd (pandas/read_csv "./resources/data/tips.csv"))

(def tip-pd (py/get-item tips-pd :tip))

(def idx-pd (py.- (pandas/Categorical (py/get-item tips-pd :day) :categories ["Thur" "Fri", "Sat" "Sun"]) codes))


(def inference-cg (py/with
                   [model (py. pm Model)]
                   (let [n-groups 4
                         mu (py. pm Normal "mu" :mu 0 :sd 10 :shape n-groups)
                         sigma (py. pm HalfNormal "sigma" :sd 10 :shape n-groups )
                         y (py. pm Normal "y"
                                :mu (py/get-item mu idx-pd)
                                :sd (py/get-item sigma idx-pd)
                                :observed tip-pd)
                         trace (py. pm sample 1000 :random_seed 123)]
                     {:trace  trace
                      :model model})))

(with-show(az/plot_trace (:trace inference-cg)))







;; and now without pandas

(def tips-raw (ds/->dataset "./resources/data/tips.csv"))

(def tips-ds
  (-> tips-raw
      (ds-pipe/string->number "day" )
      (ds-pipe/->datatype "day" :int32)))


(def inference-cg2
  (let [n-groups (count (set (tips-ds "day")))
        tip (tips-ds "tip")
        idx (tips-ds "day")]
    (py/with
     [model (py. pm Model)]
     (let [mu    (pm/Normal "mu"
                            :mu 0
                            :sd 10
                            :shape n-groups)
           sigma (pm/HalfNormal "sigma"
                                :sd 10
                                :shape n-groups)
           y     (pm/Normal "y"
                            :mu (py/get-item mu idx)
                            :sd (py/get-item sigma idx)
                            :observed tip)
           diffs (doseq [i (range n-groups)
                         j (range n-groups)
                         :when (< i j)]
                   (pm/Deterministic
                    (str "mu" i " - mu" j)
                    (operator/sub (py/get-item mu i)
                                  (py/get-item mu j))))
           trace (pm/sample 1000 :random_seed 123)]
       {:trace  trace
        :model model}))))


(with-show
  (az/plot_forest (:trace inference-cg2)))


(def n-samples (repeat 3 30))
(def g-samples (repeat 3 18))

(def group-idx (repeat  (count n-samples)))


(defn sim-data-h
  [n-samples g-samples]
  (vec (mapcat (fn [g n] (concat (repeat g 1) (repeat (- n g) 0))) g-samples n-samples)))
(sim-data-h n-samples g-samples)

(defn build-group-idx-h [n-samples]
  (vec (mapcat (fn [i el] (repeat el i)) (range (count n-samples)) n-samples)))

#_(def inference-h
  (let [{:keys [group-idx data]} (sim-data-h n-samples g-samples)
        n-groups (count (set group-idx))]
    (py/with
     [model (pm/Model)]
     (let [mu    (pm/Beta "mu" 1.0 1.0)
           kappa (pm/HalfNormal "kappa" 10)
           theta (pm/Beta "theta"
                          :alpha (operator/mul mu kappa)
                          :beta (operator/mul kappa (operator/sub 1.0 mu))
                          :shape n-groups)
           y     (pm/Bernoulli "y"
                               :p (py/get-item theta group-idx)
                               :observed data)
           trace (pm/sample 2000 :random_seed 123)]
       {:trace  trace
        :model model}))))

(with-show (az/plot_trace (:trace inference-h)))

(az/summary (:trace inference-h))

(defn run-inference-h
  [n-samples g-samples]
  (let [group-idx (build-group-idx-h n-samples)
       data (sim-data-h n-samples g-samples)]
    (py/with
     [model (pm/Model)]
     (let [mu    (pm/Beta "mu" 1.0 1.0)
           kappa (pm/HalfNormal "kappa" 10)
           theta (pm/Beta "theta"
                          :alpha (operator/mul mu kappa)
                          :beta (operator/mul kappa (operator/sub 1.0 mu))
                          :shape (count (set group-idx)))
           y     (pm/Bernoulli "y"
                               :p (py/get-item theta group-idx)
                               :observed data)
           (pm/p)
           trace (pm/sample 2000 :random_seed 123)]
       {:trace  trace
        :model model}))))

(def inference-h
  (run-inference-h n-samples g-samples))

(let [g-samples [18 3 3]
      {:keys [trace]} (run-inference-h n-samples g-samples)]
  (with-show (az/plot_trace trace))
  (az/summary trace))

(with-show (az/plot_joint (:trace inference-h) ))

(with-show (az/plot_ppc (az/from)(:trace inference-h)
                        :var_names ["theta"]
                        ))

(let [y-ppc-t (pm/sample_posterior_predictive (:trace inference-h)
                                              100
                                              (:model inference-h)
                                              :random_seed 123)
      y-pred-t (az/from_pymc3 :trace (:trace inference-h)
                              :posterior_predictive y-ppc-t)]
  (with-show (az/plot_ppc y-pred-t)))

;; TODO don't know how to make this "theta" prior yet.
;; can I use the theta vars for this?
;; or should I create new Deterministic var based on kappa and mu?
