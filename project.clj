(defproject book-bayesian-analysis-with-python "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.10.0"]
                 [cnuernber/libpython-clj "1.33"]
                 [metasoarous/oz "1.6.0-alpha5"]
                 [techascent/tech.ml.dataset "1.68"]]
  :repl-options {:init-ns book-bayesian-analysis-with-python.core})
