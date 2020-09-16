(ns book-bayesian-analysis-with-python.2
  (:require [libpython-clj.require :refer [require-python]]
            [libpython-clj.python :as py :refer [py. py.- py.. get-item]]
            [book-bayesian-analysis-with-python.utils :refer
             [with-show group-by-columns-and-aggregate plot-posterior-predictive-check ]]
            [oz.core :as oz]
            oz.server
            [tech.ml.dataset :as ds]
            [tech.ml.dataset.pipeline :as ds-pipe]
            clojure.tools.deps.alpha.repl))



(require-python '[builtins :as python]
                '[pymc3 :as pm :bind-ns]
                '[numpy :as np :bind-ns]
                'numpy.ma
                '[pymc3.math :as pm-math :bind-ns]
                '[arviz :as az]
                '[pandas :as pd]
                '[scipy.stats :as stats :refer [bernoulli beta]]
                'operator)


(python/help pm)
(assert (= 11 (np/dot [1 2 ] [ 3 4])))
np/nan
(py.- pm "__version__")
(py. bernoulli rvs :p 0.4 :size 5)

(require '[tech.ml.dataset.column :as ds-col])
(let [arr [true nil false nil]
      ds (tech.ml.dataset/name-values-seq->dataset  [["boolean" arr]])]
  (println ds)
  #_(assert (= #{1 3} (ds-col/missing (ds "boolean"))))
  (ds-col/missing (ds "boolean")))

(require '[tech.v2.tensor :as dtt])

(dtt/->tensor [1 2 np/nan])







;; (require-python '([scipy.stats :as stats :refer [bernoulli]]))

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

(def y-ppc-t (pm/sample_posterior_predictive (:trace inference-h)
                                              100
                                              (:model inference-h)
                                              :random_seed 123))
(def y-pred-t (az/from_pymc3 :trace (:trace inference-h)
                             :posterior_predictive y-ppc-t))
#_((with-show
     (az/plot_ppc y-pred-t :var_names ["kappa"])))

(py/get-item (:trace inference-h) "theta")

(with-show (az/plot_ppc (:trace inference-h) :var_names ["kappa"]))

;; TODO don't know how to make this "theta" prior yet.
;; can I use the theta vars for this?
;; or should I create new Deterministic var based on kappa and mu?


(defn run-inference-h2
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
           prior_theta
           (pm/Beta "prior_theta"
                    :alpha (operator/mul mu kappa)
                    :beta (operator/mul kappa (operator/sub 1.0 mu)))
           trace (pm/sample 2000 :random_seed 123)]
       {:trace  trace
        :model model}))))

(def inference-h2 (run-inference-h2 n-samples g-samples))

(def y-ppc-t-h2 (pm/sample_posterior_predictive (:trace inference-h2)
                                             50
                                             (:model inference-h2)
                                             :random_seed 123
                                             :var_names ["mu" "kappa" "prior_theta" ]
                                            ))


(def y-pred-t-h2 (az/from_pymc3 :trace (:trace inference-h2)
                             :posterior_predictive y-ppc-t-h2))

(py/get-item y-pred-t-h2 "posterior_predictive")


(with-show (az/plot_ppc y-pred-t-h2 :var_names ["kappa"]
                        ))


(map (fn [mu kappa] (py. (stats/beta (* mu kappa)
                                     (* kappa (- 1 mu)))
                         pdf (np/linspace 0 1 100)))
     (y-ppc-t-h2 "mu") (y-ppc-t-h2 "kappa"))


(def prior-theta-lines (mapcat (fn [mu kappa]
                             (for [x (np/linspace 0 1 40)]
                               {:group (str "kappa: " kappa ", mu " mu)
                                        ;:kappa kappa
                                        ;:mu mu
                                :x x
                                :pdf (py. (stats/beta (* mu kappa)
                                                      (* kappa (- 1 mu)))
                                          pdf x)}))
                           (y-ppc-t-h2 "mu") (y-ppc-t-h2 "kappa")))

(take 5 prior-theta-lines)

(oz/view!
 {:data {:values prior-theta-lines}
  :mark :line
  :encoding {:x {:field :x :type :quantitative}
             :y {:field :pdf :type :quantitative}
             :detail {:field :group :type :nominal}
             :opacity {:value 0.5}
             :size {:value 0.6}}})

(def prior-theta-lines2
  (mapcat (fn [mu kappa]
            (for [x (np/linspace 0 1 40)]
              {:group (str "kappa: " kappa ", mu " mu)
                                        ;:kappa kappa
                                        ;:mu mu
               :x x
               :pdf (py. (stats/beta (* mu kappa)
                                     (* kappa (- 1 mu)))
                         pdf x)}))
          (take 40 (py/get-item (:trace inference-h2) "mu"))
          (take 40 (py/get-item (:trace inference-h2) "kappa"))))

(py/get-item (:trace inference-h2) "mu") 

(take 5 prior-theta-lines2)
(count prior-theta-lines2)

(oz/view!
 {:data {:values prior-theta-lines2}
  :mark :line
  :encoding {:x {:field :x :type :quantitative}
             :y {:field :pdf :type :quantitative}
             :detail {:field :group :type :nominal}
             :opacity {:value 0.5}
             :size {:value 0.6}}})

(with-show (az/plot_pair (:trace inference-h2) :kind :kde :var_names ["mu" "kappa"]))




;; one more example

(def chem-shifts-theo-exp
  (ds/->dataset "./resources/data/chemical_shifts_theo_exp.csv"))


(def chem-shifts-theo-exp
  (-> (ds/->dataset "./resources/data/chemical_shifts_theo_exp.csv")
      (ds-pipe/string->number "aa")
      (ds-pipe/->datatype "aa" :int32)))

(require '[tech.v2.datatype.functional :as dfn])
(def diff (dfn/- (chem-shifts-theo-exp "theo") (chem-shifts-theo-exp "exp")))



(def aa->indices
  (->> chem-shifts-theo-exp
       ds/->flyweight
       (map #(get % "aa"))
       set
       (into {} (map-indexed (fn [i e] [e i])) )
       ))

(defn add-diff [df]
  (map (fn [{:strs [theo exp] :as m}]
         (assoc m :diff (- theo exp))) df))

(defn pipeline-nh [df ]
  (->> df
       ds/->flyweight
       add-diff
       (map #(update % "aa" aa->indices))
       ))

(pipeline-nh chem-shifts-theo-exp)


;; TODO see if this is easier to do with meander. I don't think
;; dataset is a good solution yet. Doesn't feel like finished thing.

(def groups-nh (count aa->indices))

(def idx-nh
  (->> (pipeline-nh chem-shifts-theo-exp)
       (mapv #(get % "aa"))))

(def data-nh
  (->> (pipeline-nh chem-shifts-theo-exp)
       (mapv :diff)))



(def run-inference-nh
    (py/with
     [model (pm/Model)]
     (let [mu    (pm/Normal "mu" :mu 0 :sd 10 :shape groups-nh)
           sigma (pm/HalfNormal "sigma" :sd 10 :shape groups-nh)
           y     (pm/Normal "y"
                            :mu (py/get-item mu idx-nh)
                            :sd (py/get-item sigma idx-nh)
                            :observed data-nh)
           trace (pm/sample 2000 :random_seed 123)]
       {:trace  trace
        :model model})))



(def run-inference-h
  (py/with
   [model (pm/Model)]
   (let [
         mu-mu (pm/Normal "mu-mu" :mu 0 :sd 10)
         sigma-mu (pm/HalfNormal "sigma-mu" :sd 10)

         mu    (pm/Normal "mu" :mu mu-mu :sd sigma-mu :shape groups-nh)
         sigma (pm/HalfNormal "sigma" :sd 10 :shape groups-nh)
         y     (pm/Normal "y"
                          :mu (py/get-item mu idx-nh)
                          :sd (py/get-item sigma idx-nh)
                          :observed data-nh)
         trace (pm/sample 2000 :random_seed 123)]
     {:trace  trace
      :model model})))

;; TODO find some way to build a macro/function that takes the name of a symbol and adds
;; it as the first parameter instead of me having to type the symbol AND the string.


(with-show
  (az/plot_forest (mapv :trace [run-inference-h run-inference-nh])
                  :var_names "mu"
                  :combined false
                  :colors "cycle"))


(def run-inference-h-swapped-order ; gives error
  (py/with
   [model (pm/Model)]
   (let [
         mu    (pm/Normal "mu" :mu mu-mu :sd sigma-mu :shape groups-nh)

         mu-mu (pm/Normal "mu-mu" :mu 0 :sd 10)
         sigma-mu (pm/HalfNormal "sigma-mu" :sd 10)

         sigma (pm/HalfNormal "sigma" :sd 10 :shape groups-nh)
         y     (pm/Normal "y"
                          :mu (py/get-item mu idx-nh)
                          :sd (py/get-item sigma idx-nh)
                          :observed data-nh)
         trace (pm/sample 2000 :random_seed 123)]
     {:trace  trace
      :model model})))

;; TODO create something in which I give the model, where the order doesn't matter.
;; so some map like data structure. either some map with maybe some plumbing liek features, or maybe some hiccup syntax or even some datalog syntax. If you'd have a datastructure you could just add components to the model programmatically. If it is non-ordered you don\t have to think about order, let the program figure it out. Error if not possible. Should probably be a dag. 
 
 
;; exercises

(def data (py. bernoulli rvs :p 0.35 :size 4))
(def res (py/with [model (pm/Model)]
                (pm/Bernoulli "y"
                              :p (pm/Beta "theta" :alpha 1 :beta 1)
                              ;; :p (pm/Uniform "theta" :lower -2 :upper 1)
                              :observed data)
                (hash-map :trace (pm/sample 1000 :random_seed 123)
                          :model model)))
;; beta 1 1 and uniform same speed
;; uniform -2 1 gives error. Logical since p of bernoulli must lie in [0, 1]


;;2:  coal mining disaster
;; https://docs.pymc.io/notebooks/getting_started.html#Case-study-2:-Coal-mining-disasters

;; disaster_data = pd.Series([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,

(def python-code-coal-disaster
  "import pandas as pd
import numpy as np
import pymc3 as pm

disaster_data = pd.Series([4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                           3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                           2, 2, 3, 4, 2, 1, 3, np.nan, 2, 1, 1, 1, 1, 3, 0, 0,
                           1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                           0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                           3, 3, 1, np.nan, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                           0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])
years = np.arange(1851, 1962)

with pm.Model() as disaster_model:

    switchpoint = pm.DiscreteUniform('switchpoint', lower=years.min(), upper=years.max(), testval=1900)

    # Priors for pre- and post-switch rates number of disasters
    early_rate = pm.Exponential('early_rate', 1)
    late_rate = pm.Exponential('late_rate', 1)

    # Allocate appropriate Poisson rates to years before and after current
    rate = pm.math.switch(switchpoint >= years, early_rate, late_rate)

    disasters = pm.Poisson('disasters', rate, observed=disaster_data)

with disaster_model:
    trace = pm.sample(10000)
")

(py/run-simple-string python-code-coal-disaster);; works. values are imputed. 


(def disaster_data [4, 5, 4, 0, 1, 4, 3, 4, 0, 6, 3, 3, 4, 0, 2, 6,
                    3, 3, 5, 4, 5, 3, 1, 4, 4, 1, 5, 5, 3, 4, 2, 5,
                    2, 2, 3, 4, 2, 1, 3, nil, 2, 1, 1, 1, 1, 3, 0, 0,
                    1, 0, 1, 1, 0, 0, 3, 1, 0, 3, 2, 2, 0, 1, 1, 1,
                    0, 1, 0, 1, 0, 0, 0, 2, 1, 0, 0, 0, 1, 1, 0, 2,
                    3, 3, 1, nil, 2, 1, 1, 1, 1, 2, 4, 2, 0, 0, 1, 4,
                    0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1])

(def disaster-data-nonil (filterv identity disaster_data))

(def years (np/arange 1851, 1962))

(with-show
  (matplotlib.pyplot/plot years disaster_data "o" :markersize 8))


(defn point-spec [x-ary y-ary]
  {:data {:values (map #(zipmap [:x :y] %&) x-ary y-ary)}
   :mark :point 
   ;; :mark {:type :point :tooltip true}
   :encoding {:x {:field :x :type :quantitative :scale {:zero false}}
              :y {:field :y :type :quantitative}}})
(defn point-spec-tooltip [x-ary y-ary]
  {:data {:values (map #(zipmap [:x :y] %&) x-ary y-ary)}
   ;; :mark :point 
   :mark {:type :point :tooltip true}
   :encoding {:x {:field :x :type :quantitative :scale {:zero false}}
              :y {:field :y :type :quantitative}}})

(oz/start-server!)
(oz/view!
 (point-spec years disaster_data))
(oz.server/stop-web-server!)


(require '[tech.v2.tensor :as dtt])

(dtt/->tensor [1 2 np/nan])


;; 1: just use 1 rate
;; 2: use 2 rates fixed switchpoint
;; 3: use 2 rates floating switchpoint
;; 4: include nan values instead of flitering out. 
(def coal-mining-inference 
  (py/with
   [model (pm/Model)]
   (let [
         ;;switchpoint (pm/DiscreteUniform "t" :lower 1851 :upper 1962)
         ;; r-early (pm/HalfNormal "r-early" :sigma 10)
         r-early (pm/Exponential "r-early" :lam 1)
         ;;r-late (pm/HalfNormal "r-late"  :sigma 10)
         ;; rate  (py.. pm math switch (>= switchpoint years) r-early r-late)
         ;; switch (py.. pm math switch )

         disasters (pm/Poisson "disasters" :mu r-early
                               :observed
                               #_(->> disaster_data (filterv identity));does work
                               ;; disaster_data doesn't work
                               ;;(np/array disaster_data) doesn't work
                               #_(mapv (fn [e] (if e e np/nan)) disaster_data)
                               ;;[1 2 2 2 ] ; works
                               ;;[1 2 2 nil ] ; doesn't work
                               ;; [1 2 2 np/nan] ; doesn't work
                               ;; (dtt/->tensor [1 2 2 np/nan]) ;; doesn't work
                               ;; (py/->numpy (dtt/->tensor [1 2 2 np/nan 4]))
                               ;; (pd/Series [1 2 2]); works
                               ;; (pd/Series [1 2 2 nil]); works
                               (pd/Series disaster_data)
                               ;; ((tech.ml.dataset/name-values-seq->dataset
                               ;; [[:nm [1 2 3]]]) :nm) ; doesn't work

                               ;; (py. (py.- np ma) array [1 2 2 2] :mask [ 0 1 0 0 ]); works
                               ;; (py. (py.- np ma) array [1 nil 2 2] :mask [ 0 1 0 0 ]);  works! 

                               #_(py. (py.- np ma) masked_values
                                    :x [1 nil 2 2]
                                    :value nil);  works!
                               ;; (numpy.ma/masked_values [1 nil 2 2] nil)
                               )


         trace (pm/sample 1000)
         ]
         {:trace  trace
          :model model})))
;; NOTE so either use pd.Series, or a masked array! 

(with-show (az/plot_trace (:trace coal-mining-inference)))
(with-show (az/plot_forest (:trace inference-cg)))

(-> coal-mining-inference
    :trace
    (py/get-item "r-early"))

((:trace coal-mining-inference) "r-early")


(py/att-type-map (:trace coal-mining-inference))


(def coal-mining-inference2
  (py/with
   [model (pm/Model)]
   (let [
         ;;switchpoint (pm/DiscreteUniform "t" :lower 1851 :upper 1962)
         ;; r-early (pm/HalfNormal "r-early" :sigma 10)
         r-early (pm/Exponential "r-early" :lam 1)
         ;;r-late (pm/HalfNormal "r-late"  :sigma 10)
         ;; rate  (py.. pm math switch (>= switchpoint years) r-early r-late)
         ;; switch (py.. pm math switch )

         disasters (pm/Poisson "disasters" :mu r-early :observed (->> disaster_data
                                                                      (filterv identity)))

         trace (pm/sample 1000)
         ]
     {:trace  trace
      :model model})))

(trace
 r-early (pm/Exponential :lam 1)
 disasters (pm/Poisson :mu r-early :observed disaster_data))

(defmacro quick-trace
  [& body]
   
  (let [
        bindings (partition 2 body)
         bindings' (mapcat (fn [[symb ls]]
                             [symb (concat (list(first ls) (name symb)) (rest ls))]) bindings)]
     (println bindings')
     `(py/with
       [_# (pm/Model)]
       (let [~@bindings']
         (pm/sample 2000)))))

(macroexpand-1 '(quick-trace
                 r-early (pm/Exponential :lam 1)
                 disasters (pm/Poisson :mu r-early :observed disaster-data-nonil)))
;; => (libpython-clj.python/with [___68826__auto__ (pymc3/Model)] (clojure.core/let [r-early (pm/Exponential "r-early" :lam 1) disasters (pm/Poisson "disasters" :mu r-early :observed disaster-data-nonil)] (pymc3/sample 1000)))



(def trace-coal-1
  (quick-trace
   r-early (pm/Exponential :lam 1)
   disasters (pm/Poisson :mu r-early :observed (pd/Series disaster_data))))

(with-show (az/plot_trace  trace-coal-1))


(quick-trace
 r-early (pm/Exponential :lam 1)
 disasters (pm/Poisson :mu r-early :observed (pd/Series disaster_data)))



(defn oz-export-png
  [spec filename]
                                        ; this allows us to call a private function
  (#'oz/vg-cli
   {:spec spec
    :format :png
    :return-output? false
    :output-filename filename}))

(comment
  (oz-export-png (point-spec years disaster_data) "oz-tmp.png")
  )

(require '[clojure.java.shell :as sh])
(defn oz-quick-view! [spec]
     (oz-export-png spec "oz-tmp.png")
     (sh/sh "open" "oz-tmp.png"))

(oz-quick-view! (point-spec years disaster_data))
(oz-quick-view! (point-spec (map (partial + 100) years) disaster_data))

(defn oz-quick-svg! [spec]
  (#'oz/vg-cli
   {:spec spec
    :format :svg
    :return-output? false
    :output-filename "oz-tmp.svg"})
  #_(sh/sh "open" "oz-tmp.svg")
  ;; the spacebar quick look application in finder
  (sh/sh "qlmanage" "-p" "oz-tmp.svg"))

(oz-quick-svg! (point-spec years disaster_data))
#_(oz-quick-svg! (point-spec-tooltip years disaster_data))
;; cause tooltip doesn't work with svg or png anyway. 



;; 1: just use 1 rate
;; 2: use 2 rates fixed switchpoint
;; 3: use 2 rates floating switchpoint
;; 4: include nan values instead of flitering out. 

(def coal-mining-inference-2 ;; doesn't work yet. :/
  (py/with
   [model (pm/Model)]
   (let [switchpoint (pm/DiscreteUniform "switchpoint" :lower 1851 :upper 1962 :testval 1900)
         r-early (pm/Exponential "r-early" :lam 1)
         r-late  (pm/Exponential "r-late"  :lam 1)
         ;; rate (py. (py. pm math) switch (np/greater_equal switchpoint years) r-early r-late)
         rate (pm-math/switch (pm-math/ge switchpoint years) r-early r-late)
         disasters (pm/Poisson "disasters" :mu rate
                               :observed
                               (pd/Series disaster_data))]
         (pm/sample 2000))))

(quick-trace
  switchpoint (pm/DiscreteUniform :lower 1851 :upper 1962 :testval 1900)
  r-early (pm/Exponential :lam 1)
  r-late  (pm/Exponential :lam 1)
  rate (pm-math/switch (pm-math/ge switchpoint years) r-early r-late) ;; rate?
  disasters (pm/Poisson :mu rate
                        :observed (pd/Series disaster_data))) ;; doesn't work

(defmacro quick-trace2
  [& body]
  
  (let [
        bindings (partition 2 body)
        bindings' (mapcat (fn [[symb ls]]
                            [symb (concat (list(first ls) :self (name symb)) (rest ls))]) bindings)]
    (println bindings')
    `(py/with
      [_# (pm/Model)]
      (let [~@bindings']
        (pm/sample 2000)))))


(quick-trace2
 switchpoint (pm/DiscreteUniform :lower 1851 :upper 1962 :testval 1900)
 r-early (pm/Exponential :lam 1)
 r-late  (pm/Exponential :lam 1)
 rate (pm-math/switch (pm-math/ge switchpoint years) r-early r-late) ;; rate?
 disasters (pm/Poisson :mu rate
                       :observed (pd/Series disaster_data))) ;; doesn't work


(def coal-mining-final
  (quick-trace
   switchpoint (pm/DiscreteUniform :lower 1851 :upper 1962 :testval 1900)
   r-early (pm/Exponential :lam 1)
   r-late  (pm/Exponential :lam 1)
   disasters (pm/Poisson
              :mu (pm-math/switch (pm-math/ge switchpoint years)
                                  r-early
                                  r-late)
              :observed (pd/Series disaster_data)))) ;; works!!
;; NOTE quick-trace breaks down when you put in elements that don't take a name as a second arg. 

(with-show (az/plot_trace  coal-mining-final))

(def coal-mining-final2 ;; don't know if works. plot dosn't. Works, but creates separate rv for each entry of years :/
  (quick-trace
   switchpoint (pm/DiscreteUniform :lower 1851 :upper 1962 :testval 1900)
   r-early (pm/Exponential :lam 1)
   r-late  (pm/Exponential :lam 1)
   rate (pm/Deterministic (pm-math/switch (pm-math/ge switchpoint years) r-early r-late))
   disasters (pm/Poisson :mu rate
                         :observed (pd/Series disaster_data)))) 
(with-show (az/plot_trace  coal-mining-final2)) ;; doesm't work. more entries for years , than max generated plot count. 









;;  --------------------   tips exercises



(defmacro quick-trace4
  [& body]
  ;; TODO can I add something that needs to be passed back besides trace, like prior predictive? 
  (let [bindings (->> (partition 2 body)
                      (mapcat (fn [[symb [dist & dist-args]]]
                                [symb `(~dist ~(name symb) ~@dist-args)]
                                #_[symb (concat (list dist (name symb))
                                                dist-args)])))]
    (println bindings)
    `(py/with
      [_# (pm/Model)]
      (let [~@bindings
            trace# (pm/sample 2000)]
        {:trace trace#
         ;; :sample-dataset (trace->dataset trace#)
         :prior-pred-sample (pm/sample_prior_predictive)
         :posterior-pred-sample (pm/sample_posterior_predictive trace# :samples 1000)}))))


kebab

(clojure.tools.deps.alpha.repl/add-lib 'camel-snake-kebab {:mvn/version "0.4.1"})

(camel-snake-kebab.core/->kebab-case  {"aBoat" 5})


 (transform-keys csk/->kebab-case-keyword {"firstName" "John", "lastName" "Smith"})

(camel-snake-kebab.core/->kebab-case-keyword "aBoat"  )


(def tips (ds/->dataset "resources/data/tips.csv"))

(-> tips
    ((zipmap ( ds/column-names tips))))

(zipmap (ds/column-names tips)
        (map camel-snake-kebab.core/->kebab-case-keyword (ds/column-names tips)))




(defn tips-ds []
  (let [col-names  (ds/column-names tips)
        rename-map (zipmap col-names (map camel-snake-kebab.core/->kebab-case-keyword col-names))]
    (-> tips
        (ds/rename-columns rename-map)
        (ds-pipe/string->number :day )
        (ds-pipe/->datatype :day :int32))))


(def inference-cg2
  (let [n-groups (count (set ((tips-ds) :day)))
        tip ((tips-ds) :tip)
        idx ((tips-ds) :day)]
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


(ds/unique-by-column "day" (tips-ds))
(ds-col/unique ((tips-ds) "day"))

 
(defmacro quick-trace
  [& body]
    (println body)
    `(py/with
      [model# (pm/Model)]
      (let [~@body
            trace# (pm/sample 2000)]
        {:trace trace#
         :model model#
         ;;:samples (-> trace# py/->jvm ds/->dataset)
         ;; :prior-pred-sample (pm/sample_prior_predictive)
         ;; :posterior-pred-sample (pm/sample_posterior_predictive trace# :samples 1000)
         })))

;; (macroexpand-1 '(quick-trace a (pm/Normal "a" :mu 0 :sigma 1)))


(def tips-start
  (let [n-groups (count (set ((tips-ds) :day)))
        day-idx ((tips-ds) :day)]
    (quick-trace
     mu    (pm/Normal "mu"
                      :mu 0
                      :sd 10
                      :shape n-groups)
     sigma (pm/HalfNormal "sigma"
                          :sd 10
                          :shape n-groups)
     y     (pm/Normal "y"
                      :mu (get-item mu day-idx)
                      :sd (get-item sigma day-idx)
                      :observed ((tips-ds) :tip))
     diffs (doseq [i (range n-groups)
                   j (range n-groups)
                   :when (< i j)]
             (pm/Deterministic
              (str "mu" i " - mu" j)
              (operator/sub (get-item mu i)
                            (get-item mu j)))))))


(let [prior (pm/sample_prior_predictive :model (tips-start :model))
      posterior-pred (pm/sample_posterior_predictive :trace (tips-start :trace)
                                                     :model (tips-start :model) :samples 100)
      inf-obj (az/from_pymc3 :trace (tips-start :trace)
                             :prior prior
                             :posterior_predictive posterior-pred)
      ]
  (with-show (az/plot_ppc inf-obj))
  )


(let [{:keys [model trace]} tips-start
      prior (pm/sample_prior_predictive )])
(az/from_pymc3 :trace (:trace tips-start)  :posterior_predictive (:posterior-pred-sample res))
(tips-start :trace)

;; (let [{:keys [trace model]} inference-gt
;;       y-ppc-t (pm/sample_posterior_predictive trace 100 model :random_seed 123)
;;       y-pred-t (az/from_pymc3 :trace trace :posterior_predictive y-ppc-t)]
;;   (with-show
;;     (az/plot_ppc y-pred-t)
;;     (matplotlib.pyplot/xlim 40 70)))



;; 2.5. Modify the tips example to make it robust to outliers. Try with one shared V for all groups and also with one V  per group. Run posterior predictive checks to assess these three models.

(with-show (az/plot_kde ((tips-ds) :tip) :rug true))
;; yes outliers, so student t is appropriate


(def tips-2-5-shared
  (let [n-groups (count (set ((tips-ds) :day)))
        day-idx ((tips-ds) :day)]
    (quick-trace
     mu    (pm/Normal "mu"
                      :mu 0
                      :sd 10
                      :shape n-groups)
     sigma (pm/HalfNormal "sigma"
                          :sd 10
                          :shape n-groups)
     nu    (pm/Exponential "nu"
                           :lam (/ 1 30))
     y     (pm/StudentT "y"
                      :mu (get-item mu day-idx)
                      :sd (get-item sigma day-idx)
                      :nu nu
                      :observed ((tips-ds) :tip))
     diffs (doseq [i (range n-groups)
                   j (range n-groups)
                   :when (< i j)]
             (pm/Deterministic
              (str "mu" i " - mu" j)
              (operator/sub (get-item mu i)
                            (get-item mu j)))))))

(def tips-2-5-per-group
  (let [n-groups (count (set ((tips-ds) :day)))
        day-idx ((tips-ds) :day)]
    (quick-trace
     mu    (pm/Normal "mu"
                      :mu 0
                      :sd 10
                      :shape n-groups)
     sigma (pm/HalfNormal "sigma"
                          :sd 10
                          :shape n-groups)
     nu    (pm/Exponential "nu"
                           :lam (/ 1 30)
                           :shape n-groups)
     y     (pm/StudentT "y"
                        :mu (get-item mu day-idx)
                        :sd (get-item sigma day-idx)
                        :nu (get-item nu day-idx)
                        :observed ((tips-ds) :tip))
     diffs (doseq [i (range n-groups)
                   j (range n-groups)
                   :when (< i j)]
             (pm/Deterministic
              (str "mu" i " - mu" j)
              (operator/sub (get-item mu i)
                            (get-item mu j)))))))


;; ;; load from utils instead
;;   (defn plot-posterior-predictive-check [{:keys [trace model]} {:keys [xlim]}]
;;            (let [prior (pm/sample_prior_predictive :model model)
;;                  posterior-pred (pm/sample_posterior_predictive :trace trace
;;                                                                 :model model :samples 100)
;;                  az-inf-obj (az/from_pymc3 :trace trace
;;                                            :prior prior
;;                                            :posterior_predictive posterior-pred)]
;;              (with-show
;;                (az/plot_ppc az-inf-obj)
;;                (when xlim
;;                  (apply matplotlib.pyplot/xlim xlim)))))


(plot-posterior-predictive-check tips-start)
;; observed mean smaller than predicted mean
(az/summary (tips-start :trace))
;; =>             mean     sd  hpd_3%  hpd_97%  ...  ess_sd  ess_bulk  ess_tail  r_hat
;; mu[0]      2.770  0.159   2.477    3.067  ...  4876.0    4889.0    3351.0    1.0
;; mu[1]      2.735  0.248   2.255    3.185  ...  4614.0    4662.0    3008.0    1.0
;; mu[2]      2.990  0.177   2.659    3.323  ...  5097.0    5206.0    3073.0    1.0
;; mu[3]      3.254  0.147   2.978    3.521  ...  5219.0    5217.0    3018.0    1.0
;; sigma[0]   1.264  0.120   1.046    1.487  ...  4446.0    4884.0    3021.0    1.0
;; sigma[1]   1.090  0.199   0.768    1.475  ...  3503.0    4219.0    2824.0    1.0
;; sigma[2]   1.654  0.126   1.424    1.884  ...  5739.0    6316.0    3177.0    1.0
;; sigma[3]   1.254  0.104   1.057    1.436  ...  4679.0    4892.0    3260.0    1.0
;; mu0 - mu1  0.035  0.294  -0.538    0.556  ...  2036.0    4606.0    2888.0    1.0
;; mu0 - mu2 -0.219  0.241  -0.669    0.236  ...  3101.0    5311.0    3221.0    1.0
;; mu0 - mu3 -0.484  0.219  -0.886   -0.067  ...  5022.0    5561.0    3225.0    1.0
;; mu1 - mu2 -0.254  0.300  -0.841    0.286  ...  3696.0    4943.0    3228.0    1.0
;; mu1 - mu3 -0.519  0.291  -1.104   -0.013  ...  3980.0    4792.0    3178.0    1.0
;; mu2 - mu3 -0.265  0.229  -0.679    0.170  ...  3925.0    4937.0    3304.0    1.0

(plot-posterior-predictive-check tips-2-5-shared)
(az/summary (tips-2-5-shared :trace))
;; =>             mean     sd  hpd_3%  hpd_97%  ...  ess_sd  ess_bulk  ess_tail  r_hat
;; mu[0]      2.574  0.159   2.287    2.879  ...  4479.0    4499.0    3382.0    1.0
;; mu[1]      2.714  0.264   2.232    3.202  ...  5409.0    5533.0    2549.0    1.0
;; mu[2]      2.724  0.143   2.458    2.989  ...  4336.0    4519.0    2714.0    1.0
;; mu[3]      3.181  0.147   2.898    3.456  ...  5576.0    5658.0    2804.0    1.0
;; sigma[0]   1.035  0.137   0.792    1.298  ...  3888.0    4000.0    3243.0    1.0
;; sigma[1]   0.988  0.213   0.640    1.396  ...  4781.0    5533.0    2931.0    1.0
;; sigma[2]   1.126  0.151   0.845    1.406  ...  2660.0    3098.0    2197.0    1.0
;; sigma[3]   1.103  0.118   0.881    1.325  ...  4195.0    4439.0    2792.0    1.0
;; nu         6.130  4.190   2.574   11.068  ...  1449.0    2676.0    1834.0    1.0
;; mu0 - mu1 -0.140  0.308  -0.685    0.468  ...  2077.0    5575.0    2856.0    1.0
;; mu0 - mu2 -0.151  0.201  -0.534    0.217  ...  2974.0    5888.0    3141.0    1.0
;; mu0 - mu3 -0.607  0.209  -0.983   -0.188  ...  5206.0    5640.0    3000.0    1.0
;; mu1 - mu2 -0.010  0.300  -0.548    0.577  ...  1719.0    5521.0    2792.0    1.0
;; mu1 - mu3 -0.467  0.305  -1.026    0.111  ...  4162.0    5683.0    3219.0    1.0
;; mu2 - mu3 -0.456  0.202  -0.839   -0.096  ...  4918.0    5806.0    3199.0    1.0


(plot-posterior-predictive-check tips-2-5-per-group {:xlim [-2 12]})
;; less difference between predictive y and observed y
(az/summary (tips-2-5-per-group :trace))
;; =>              mean      sd  hpd_3%  hpd_97%  ...  ess_sd  ess_bulk  ess_tail  r_hat
;; mu[0]       2.678   0.177   2.370    3.039  ...  3668.0    3691.0    2180.0    1.0
;; mu[1]       2.727   0.250   2.258    3.203  ...  5930.0    6231.0    3025.0    1.0
;; mu[2]       2.665   0.127   2.418    2.898  ...  4419.0    4436.0    2997.0    1.0
;; mu[3]       3.236   0.149   2.968    3.538  ...  6319.0    6339.0    2642.0    1.0
;; sigma[0]    1.157   0.151   0.857    1.428  ...  2625.0    2770.0    1984.0    1.0
;; sigma[1]    1.061   0.206   0.712    1.447  ...  4058.0    5261.0    2791.0    1.0
;; sigma[2]    0.975   0.139   0.717    1.234  ...  3856.0    4409.0    2938.0    1.0
;; sigma[3]    1.220   0.112   1.013    1.427  ...  4866.0    5208.0    2964.0    1.0
;; nu[0]      27.881  27.342   1.359   78.239  ...  3843.0    2670.0    2117.0    1.0
;; nu[1]      35.694  29.287   1.573   87.708  ...  4104.0    5869.0    3203.0    1.0
;; nu[2]       3.220   1.377   1.440    5.392  ...  1656.0    3801.0    2737.0    1.0
;; nu[3]      42.086  33.086   3.035  103.185  ...  4192.0    6665.0    3493.0    1.0
;; mu0 - mu1  -0.049   0.311  -0.670    0.509  ...  1994.0    5156.0    2563.0    1.0
;; mu0 - mu2   0.013   0.218  -0.398    0.414  ...  2221.0    3756.0    2705.0    1.0
;; mu0 - mu3  -0.558   0.231  -1.016   -0.132  ...  3866.0    4484.0    2933.0    1.0
;; mu1 - mu2   0.061   0.279  -0.480    0.576  ...  1995.0    5570.0    3283.0    1.0
;; mu1 - mu3  -0.509   0.289  -1.076    0.002  ...  4475.0    6118.0    3060.0    1.0
;; mu2 - mu3  -0.570   0.196  -0.933   -0.199  ...  5000.0    5461.0    3087.0    1.0
(let [{:keys [model trace]} tips-2-5-per-group
      prior (pm/sample_prior_predictive :model model)
      posterior-pred (pm/sample_posterior_predictive :trace trace
                                                     :model model :samples 100)
      az-inf-obj (az/from_pymc3 :trace trace
                                :prior prior
                                :posterior_predictive posterior-pred)]
  (with-show
    (az/plot_ppc az-inf-obj)
    (matplotlib.pyplot/xlim -2 12)))



(plot-posterior-predictive-check tips-2-5-per-group {:xlim [-2 15]})




;; 2.6. Compute the probability of superiority directly from the posterior (without computing Cohen's d first). You can use the pm.sample_posterior_predictive() function to take a sample from each group. Is it really different from the calculation assuming normality? Can you explain the result?

