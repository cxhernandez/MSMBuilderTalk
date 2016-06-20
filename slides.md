

% title: <div style="text-align: center; margin-left: 0em; margin-bottom: -.3em;"> <img height=300 src="http://msmbuilder.org/3.5.0/_static/logo.png"></div>
% author: Carlos X. Hernández, Matthew P. Harrigan, Muneeb M. Sultan, Brooke E. Husic
% author: Updated: Jun. 23, 2016 (msmbuilder v3.5)



---
title: Old-School Analysis of MD Data

- Analysis happens in walled gardens (Gromacs, Amber, VMD)
- Exclusively command line interfaces, C and Fortran code
- Duplication of statistical algorithms by non-experts (e.g. chemists, biologists)
- Possible code maintainability issues?

---
title: Jarvis Patrick Clustering in Gromacs
subtitle: <a href="https://github.com/gromacs/gromacs/blob/master/src/gromacs/gmxana/gmx_cluster.cpp#L502">real code in gromacs</a>

<pre class="prettyprint" data-lang="c++">

static void jarvis_patrick(int n1, real **mat, int M, int P,
real rmsdcut, t_clusters *clust) {
t_dist *row;
t_clustid *c;
int **nnb;
int i, j, k, cid, diff, max;
gmx_bool bChange;
real **mcpy = NULL;
if (rmsdcut < 0) {
rmsdcut = 10000;
}
/* First we sort the entries in the RMSD matrix row by row.
* This gives us the nearest neighbor list.
*/
</pre>

---
title: Jarvis Patrick Clustering in Gromacs (Cont.)

<pre class="prettyprint" data-lang="c++">

// Five more pages of this
// You get the idea

// Also, how do we even use this function?
static void jarvis_patrick(int n1, real **mat, int M, int P,
real rmsdcut, t_clusters *clust);
</pre>



---
title: Enter Data Science

- Machine learning is mainstream now!
- Thousands of experts are using machine learning approaches
- Well-tested, performant, and facile implementations are available
- Writing your own is not the way to go!
    * E.g: Is clustering _that_ special and MD-specific such that
    we need our own custom algorithms and implementations?

---
title: MSMBuilder3: Design

<div style="float:right; margin-top:-100px">
<img src="figures/flow-chart.png" height="600">
</div>

Builds on [scikit-learn](http://scikit-learn.org/stable/) idioms:

- Everything is a `Model`.
- Models are `fit()` on data.
- Models learn `attributes_`.
- `Pipeline()` concatenate models.
- Use best-practices (cross-validation)

<footer class="source">
<a href="http://msmbuilder.org/3.5.0/apipatterns.html">
http://msmbuilder.org/3.5.0/apipatterns.html
</a>
</footer>

---
title: Everything is a <code>Model()</code>!


<pre class="prettyprint" data-lang="python">

>>> import msmbuilder.cluster
>>> clusterer = msmbuilder.cluster.KMeans(n_clusters=4)

>>> import msmbuilder.decomposition
>>> tica = msmbuilder.decomposition.tICA(n_components=3)

>>> import msmbuilder.msm
>>> msm = msmbuilder.msm.MarkovStateModel()

</pre>

Hyperparameters go in the constructor.


<footer class="source">
Actually, everything is a <code>sklearn.base.BaseEstimator()</code>
</footer>


---
title: Models <code>fit()</code> data!

<pre class="prettyprint" data-lang="python">

>>> import msmbuilder.cluster

>>> trajectories = [np.random.normal(size=(100, 3))]

>>> clusterer = msmbuilder.cluster.KMeans(n_clusters=4, n_init=10)
>>> clusterer.fit(trajectories)

>>> clusterer.cluster_centers_

array([[-0.22340896,  1.0745301 , -0.40222902],
       [-0.25410827, -0.11611431,  0.95394687],
       [ 1.34302485,  0.14004818,  0.01130485],
       [-0.59773874, -0.82508303, -0.95703567]])


</pre>

Estimated parameters *always* have trailing underscores!


---
title: <code>fit()</code> acts on lists of sequences


<pre class="prettyprint" data-lang="python">

>>> import msmbuilder.msm

>>> trajectories = [np.array([0, 0, 0, 1, 1, 1, 0, 0])]

>>> msm = msmbuilder.msm.MarkovStateModel()
>>> msm.fit(trajectories)

>>> msm.transmat_

array([[ 0.75      ,  0.25      ],
       [ 0.33333333,  0.66666667]])

</pre>

This is different from sklearn, which uses 2D arrays.


---
title: Models <code>transform()</code> data!

<pre class="prettyprint" data-lang="python">

>>> import msmbuilder.cluster

>>> trajectories = [np.random.normal(size=(100, 3))]

>>> clusterer = msmbuilder.cluster.KMeans(n_clusters=4, n_init=10)
>>> clusterer.fit(trajectories)
>>> Y = clusterer.transform(trajectories)

[array([5, 6, 6, 0, 5, 5, 1, 6, 1, 7, 5, 7, 4, 2, 2, 2, 5, 3, 0, 0, 1, 3, 0,
        5, 5, 0, 4, 0, 0, 3, 4, 7, 3, 5, 5, 5, 6, 1, 1, 0, 0, 7, 4, 4, 2, 6,
        1, 4, 2, 0, 2, 4, 4, 5, 2, 6, 3, 2, 0, 6, 3, 0, 7, 7, 7, 0, 0, 0, 3,
        3, 2, 7, 6, 7, 2, 5, 1, 0, 3, 6, 3, 2, 0, 5, 0, 3, 4, 2, 5, 4, 1, 5,
        5, 4, 3, 3, 7, 2, 1, 4], dtype=int32)]

</pre>
Moving the data-items from one "space" / representation into another.

---
title: <code>Pipeline()</code> concatenates models!


<pre class="prettyprint" data-lang="python">

>>> import msmbuilder.cluster, msmbuilder.msm
>>> from sklearn.pipeline import Pipeline

>>> trajectories = [np.random.normal(size=(100, 3))]

>>> clusterer = msmbuilder.cluster.KMeans(n_clusters=2, n_init=10)
>>> msm = msmbuilder.msm.MarkovStateModel()
>>> pipeline = Pipeline([("clusterer", clusterer), ("msm", msm)])

>>> pipeline.fit(trajectories)
>>> msm.transmat_

array([[ 0.53703704,  0.46296296],
       [ 0.53333333,  0.46666667]])

</pre>
Data "flows" through transformations in the pipeline.

---
title: Loading Trajectories

You can use MDTraj to load your trajectory files

<pre class="prettyprint" data-lang="python">

>>> import glob
>>> import mdtraj as md

>>> filenames = glob.glob("./Trajectories/ala_*.h5")
>>> trajectories = [md.load(filename) for filename in filenames]

</pre>

<footer class="source">
Note: for big datasets, you can get fancy with <code><a href="http://mdtraj.org/latest/api/generated/mdtraj.iterload.html">md.iterload</a></code>.
</footer>


---
title: Featurization

Featurizers wrap MDTraj functions via the `transform()` function

<div style="float:right;">
<img height=225 src=figures/rama.png />
</div>


<pre class="prettyprint" style="width:75%" data-lang="python">

>>> from msmbuilder.featurizer import DihedralFeaturizer
>>> from matplotlib.pyplot import hexbin, plot

>>> featurizer = DihedralFeaturizer(
...        ["phi", "psi"], sincos=False)
>>> X = featurizer.transform(trajectories)
>>> phi, psi = np.rad2deg(np.concatenate(X).T)

>>> hexbin(phi, psi)
</pre>

<footer class="source">
<a href="http://msmbuilder.org/3.5.0/featurization.html">
http://msmbuilder.org/3.5.0/featurization.html
</a>
</footer>

---
title: Featurization (Cont.)

You can even combine featurizers with <code>FeatureSelector</code>

<pre class="prettyprint" data-lang="python">

>>> from msmbuilder.featurizer import DihedralFeaturizer, ContactFeaturizer
>>> from msmbuilder.feature_selection import FeatureSelector

>>> dihedrals = DihedralFeaturizer(
...         ["phi", "psi"], sincos=True)
>>> contacts = ContactFeaturizer(scheme='ca')
>>> featurizer = FeatureSelector([('dihedrals', dihedrals),
...                               ('contacts', contacts)])
>>> X = featurizer.transform(trajectories)

</pre>

<footer class="source">
<a href="http://msmbuilder.org/3.5.0/featurization.html">
http://msmbuilder.org/3.5.0/featurization.html
</a>
</footer>


---
title: Preprocessing

Preprocessors normalize/whiten your data


<pre class="prettyprint" data-lang="python">

>>> from msmbuilder.preprocessing import RobustScaler
>>> scaler = RobustScaler()

>>> Y = scaler.transform(X)
</pre>

This is essential when combining different featurizers!

Also check out <code>MinMaxScaler</code> and <code>StandardScaler</code>

---
title: Decomposition

Reduce the dimensionality of your data

<div style="float:right;">
<img width=275 src="http://msmbuilder.org/3.5.0/_images/tica_vs_pca.png"/>
<figcaption style="font-size: 50%; text-align: center;">
tICA finds the slowest degrees<br>of freedom in time-series data
</figcaption>
</div>

<pre class="prettyprint" style="width:75%"  data-lang="python">

>>> from msmbuilder.decomposition import tICA

>>> tica = tICA(n_components=2, lagtime=5)
>>> Y = tica.fit_transform(X)
</pre>

Also check out <code>PCA</code> and <code>SparseTICA</code>

<footer class="source">
<a href="http://msmbuilder.org/3.5.0/decomposition.html">
http://msmbuilder.org/3.5.0/decomposition.html
</a>
</footer>

---
title: Markov State Models

We offer two main flavors of MSM:

* <code>MarkovStateModel</code> - Fits a first-order Markov model to a discrete-time integer labeled timeseries.
* <code>ContinuousTimeMSM</code> - Estimates a continuous rate matrix from discrete-time integer labeled timeseries.

Each has a Bayesian version, which estimates the error associated with the model.

<footer class="source">
MarkovStateModel: <a href="http://msmbuilder.org/3.5.0/msm.html">
http://msmbuilder.org/3.5.0/msm.html
</a>
<br>
ContinuousTimeMSM: <a href="http://msmbuilder.org/3.5.0/ratematrix.html">
http://msmbuilder.org/3.5.0/ratematrix.html
</a>
</footer>

---
title: Hidden Markov Models

<div style="float: right;">
<img height=225" src="http://msmbuilder.org/3.5.0/_images/kde-vs-histogram.png"/>
<figcaption style="font-size: 50%; text-align: center;">KDE is to histogram as HMM is to MSM</figcaption>
</div>

We also offer two types of HMMs:

* <code>GaussianHMM</code> - Reversible Gaussian Hidden Markov Model L1-Fusion Regularization
* <code>VonMisesHMM</code> - Hidden Markov Model with von Mises Emissions

HMMs are great for macrostate modeling!

<footer class="source">
<a href="http://msmbuilder.org/3.5.0/hmm.html">
http://msmbuilder.org/3.5.0/hmm.html
</a>
</footer>

---
title: Cross-Validation

<pre class="prettyprint" data-lang="python">

from sklearn.cross_validation import ShuffleSplit

cv = ShuffleSplit(len(trajectories), n_iter=5, test_size=0.5)

for fold, (train_index, test_index) in enumerate(cv):
    train_data = [trajectories[i] for i in train_index]
    test_data = [trajectories[i] for i in test_index]
    model.fit(train_data)
    model.score(test_data)
</pre>

Also check out scikit-learn's <code>KFold</code>, <code>GridSearchCV</code> and <code>RandomizedSearchCV</code>.

<footer class="source">
If you'd like to see how CV can be done with MSMs, see
<a href="http://msmbuilder.org/3.5.0/gmrq.html">
http://msmbuilder.org/3.5.0/gmrq.html
</a>
</footer>


---
title: Command-line Tools

We also offer an easy-to-use CLI for the API-averse

<pre class="prettyprint" data-lang="shell">

$ msmb DihedralFeaturizer --top my_protein.pdb --trjs "*.xtc" \
    --transformed diheds --out featurizer.pkl

$ msmb tICA -i diheds/ --out tica_model.pkl \
    --transformed tica_trajs.h5 --n_components 4

$ msmb MiniBatchKMeans -i tica_trajs.h5 \
    --transformed labeled_trajs.h5 --n_clusters 100

$ msmb MarkovStateModel -i labeled_trajs.h5 \
   --out msm.pkl --lag_time 1

</pre>


<footer class="source">
<a href="http://msmbuilder.org/3.5.0/examples/Intro/Intro.cmd.html">
http://msmbuilder.org/3.5.0/examples/Intro/Intro.cmd.html
</a>
</footer>


---
title: Related Projects

We also maintain:

* [<b>Osprey</b>](http://github.com/msmbuilder/osprey)- machine learning hyperparameter optimization
* [<b>MDEntropy</b>](http://github.com/msmbuilder/mdentropy) - entropy calculations for MD data

---
title: Osprey

Fully-automated, large-scale hyperparameter optimization

<div style="text-align: center">
<img height=225 style="padding-bottom: 1em;" src="http://msmbuilder.org/osprey/development/_static/osprey.svg"/>
<figcaption>http://github.com/msmbuilder/osprey</figcaption>
</div>

<footer class="source">
Not just for MSMs!
</footer>

---
title: Osprey: Estimator

Define your model

<pre class="prettyprint" data-lang="yaml">

estimator:
  # The model/estimator to be fit.

  eval_scope: msmbuilder
  eval: |
    Pipeline([
            ('featurizer', DihedralFeaturizer(types=['phi', 'psi'])),
            ('scaler', RobustScaler()),
            ('tica', tICA(n_components=2)),
            ('cluster', MiniBatchKMeans()),
            ('msm', MarkovStateModel(n_timescales=5, verbose=False)),
    ])

</pre>

---
title: Osprey: Search Strategy

Choose how to search over your hyperparameter space

<pre class="prettyprint" data-lang="yaml">

strategy:

    name: gp  # or random, grid, hyperopt_tpe
    params:
      seeds: 50
</pre>

---
title: Osprey: Search Space

Select which hyperparameters to optimize

<pre class="prettyprint" data-lang="yaml">

search_space:
  featurizer__types:
    choices:
      - ['phi', 'psi']
      - ['phi', 'psi', 'chi1']
    type: enum

  cluster__n_clusters:
    min: 2
    max: 1000
    type: int
    warp: log # search over log-space
</pre>

---
title: Osprey: Cross-Validation

Pick your favorite cross-validator

<pre class="prettyprint" data-lang="yaml">

cv:
  name: shufflesplit # Or kfold, loo, stratifiedshufflesplit, stratifiedkfold, fixed
  params:
    n_iter: 5
    test_size: 0.5
</pre>

---
title: Osprey: Dataset Loader

Load your data, no matter what file type

<pre class="prettyprint" data-lang="yaml">

dataset_loader:
  # specification of the dataset on which to train the models.
  name: mdtraj # Or msmbuilder, numpy, filename, joblib, sklearn_dataset, hdf5
  params:
    trajectories: ./fs_peptide/trajectory-*.xtc
    topology: ./fs_peptide/fs-peptide.pdb
    stride: 100
</pre>

---
title: Osprey: Trials

Save to a single SQL-like database, run on as many clusters as you'd like*

<pre class="prettyprint" data-lang="yaml">

trials:
  # path to a database in which the results of each hyperparameter fit
  # are stored any SQL database is supported, but we recommend using
  # SQLLite, which is simple and stores the results in a file on disk.
  uri: sqlite:///osprey-trials.db
</pre>

<footer class="source">
*you'll still need to copy your data to each cluster, however
</footer>
