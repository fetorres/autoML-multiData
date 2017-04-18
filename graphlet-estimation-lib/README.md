# Parallel Parameterized Graphlet Decomposition (PGD) & Estimation Library

A fast parallel graphlet estimation library for large graphs.



## Synopsis

In short, a parameterized high performance library for estimating the counts of motifs in large sparse graphs.


#### Setup
First, you'll need to compile PGD.  

    $ cd path/to/pgd/
    $ make

Afterwards, the following should work:  

	# compute the motif counts
	./pgd -f sample_graph.csv

Currently, PGD supports only UNIX-based systems.
*PGD* has been tested on Ubuntu linux (10.10 tested) and Mac OSX (Lion tested) with gcc-mp-4.7 and gcc-mp-4.5.4

Please let us know if you run into any issues.  



#### Input file format

+ Matrix Market Coordinate Format (symmetric)  
For details see: <http://math.nist.gov/MatrixMarket/formats.html#MMformat>  

        %%MatrixMarket matrix coordinate pattern symmetric  
        4 4 6  
        2 1  
        3 1  
        3 2  
        4 1  
        4 2  
        4 3 

Note comments are denoted by `%`. First line represents `n n m` where n is the number of nodes and m is the number of edges |E|. 
For instance, the first line above is `4 4 6` and indicates the number of nodes is n=4 and number of edges is m=6.

A graph file with the extension `.mtx` is read using this strict mtx graph reader. 
Thus, if the graph file does not strictly follow the above mtx format, 
then the file extension should be changed to allow it to be read by the more flexible graph reader discussed below.


+ Edge list: These codes are designed to be as flexible as possible and accept many variations of edge lists. Note these codes may be slightly slower than the mtx reader. This is due to allowing flexible edge list formats. Hence, this reader must perform many checks to figure out the exact format of the input file, and performs any necessary preprocessing work that may be required.


    * Delimiters: The mcpack reader accepts comma, space, or tab delimited edge lists.
            
    * Comments: Comments are allowed and should be denoted with the first character of a newline as # or %

    * Weights: If an edge list contains weights on the 3rd column, they are simply ignored. A user may specify to read the weights by setting the wt parameter or by noting the graph is in fact a temporal graph.

    * Multigraph: When an edge list contains multiple edges, we simply remove the duplicate edges.

    * The edge list may also contain gaps in the vertex ids (non sequential vertex ids) and start from any positive integer. Self-loops are removed.

    * The edge list is assumed to be undirected. However, if a directed graph is given, it is simply treated as undirected.







## Estimation Methods

Exact motif counting is the default and is used as long as 
    * `-p <percent of edges/nodes to count graphlets from>` _and_ 
    * `-k <num graphlets to sample>`. Fast approximation methods are 


#### Estimation via Uniform Sampling of Edges


Use the uniform edge sampling method by setting `-p` followed by the percent of edges to use in the approximation. For instance, 

        # estimate graphlet freq dist (GFD) using 10 percent of the edges.
        ./pgd -f sample_graph.csv -p 0.1
        
Accuracy increases as the percent p increases (sample size).

Note that this method is used only if the percent is specified.


#### Estimation via Group-based Sampling
The group-based estimation method groups edges based on their maximum k-core number and samples uniformly and cyclically from each group.
To estimate the various graphlet patterns via group-based estimation, set `-e group`.

        # estimate the graphlet patterns using group-based estimation
        ./pgd -f sample_graph.csv -p 0.1 -e group

Let us note that if `-e` is not set, then `e uniform` is used as default which corresponds to estimation via uniform edge sampling.


#### Independent Graphlet Sampling

Graphlet sampling methods based on independent sampling of four vertices. 
Simply specify the number of graphlets to sample using the option `-k x' where x is a positive integer indicating the number of uniformly sampled graphlets by replacement to use.
For instance,

        ./pgd -f sample_graph.csv -k 10000

Note that default is 0.


#### Sampling with Replacement

The above estimation methods all use a sampling routine to obtain an accurate approximation of the various graphlet patterns.
The sampling method of each use sampling without replacement (by default).
To use sampling with replacement, simply set the `--replacement` flag as follows:

        # estimate the graphlet patterns using group-based estimation
        ./pgd -f sample_graph.csv -p 0.1 -e group -- replacement
        
Similarly, for uniform sampling which is the default if unspecified, then:

        # estimate the graphlet patterns using group-based estimation
        ./pgd -f sample_graph.csv -p 0.1
        
    

## Output Graphlet Quantities
The _PGD_ family of graphlet decomposition algorithms provide three types of output 
    1. Global macro statistics indicating the total frequency of each motif
    2. Local micro statistics representing the number of times each motif appears (for each edge or node in the graph)
    3. Graphlet frequency distribution
    
NOTE: The total counts for each motif is outputted to the screen by default.

#### Macro motif counts
The macro (global) graphlet counts are printed to the screen by default.
These statistics may also be saved to a file using `--macro filename.macro` where `filename.macro` is the path to save stats.

        ./pgd -f sample_graph.csv --macro sample_graph.macro
        

#### Micro motif counts
The motif counts for each edge may also be saved using the `--micro filename.micro`. 

        ./pgd -f sample_graph.csv --micro sample_graph.micro
        

#### Graphlet frequency distribution (GFD)
To output the graphlet frequency distribution, use the `--gfd filename.gfd` option.

        ./pgd -f sample_graph.csv --gfd sample_graph.gfd
    
  
    



## Advanced

#### Orderings

The PGD algorithms are easily adapted to use various ordering strategies. 
To prescribe an ordering, use the -o option with one of the following:


Ordering techniques | Description
------------------- | -------------
`deg`           | order by degree
`kcore`         | degeneracy order
`dual_deg`      | ordering defined by the sum of degrees from neighbors
`dual_kcore`    | order by the sum of core numbers from neighbors
`kcore_deg`     | order by the weight k(v)d(v)
`rand`          | uniformly random order
`natural`       | use the order given as input



##### Direction of ordering

Descending order is the default (largest to smallest). 
For ascending order (smallest to largest), simply set ``--s2l`` as follows:

    ./pgd -f sample_graph.csv --s2l

To reverse the order of the neighbors (for each node) use ``--s2l_neigh``.






Command Line Interface (CLI)
---------

You can execute PGD with ``--help`` command to see the list of options

    $ ./pgd --help
    
	=================================================================================
	Parameterized Graphlet Decomposition (PGD): A Parallel Motif Counting Library
	=================================================================================
	-f, --file,--graph              : Input GRAPH file for computing the graphlets (e.g., matrix market format, simple edge list). 
	-a, --algorithm                 : Algorithm for the GRAPHLET DECOMPOSITION. Default: exact, approximate, etc.  
	-p, --percent                   : Percent of edges/nodes to use in the estimation (accuracy increases as p increases).
	                                  Example: '-p 0.1', that is, estimate graphlet freq dist (GFD) using 10 percent of the edges.
	                                  NOTE: Percent is IGNORED if algorithm is unspecified (default) or set to '-a exact'
	-k, --sample                    : Number of uniformly sampled graphlets by replacement
	                                  NOTE that if k is set and k>0, then graphlets are sampled based on independent sampling of four vertices
	---------------------------------------------------------------------------------
	-w, --workers                   : Number of WORKERS (processing units) for the algorithm to use (default = max). 
	-b, --block_size                : Size of blocks assigned to workers (processing units), that is, 1, 64, 512, etc.  Default: -b 64
	---------------------------------------------------------------------------------
	-o, --ordering                  : Strategy used to determine the order in which the edge/node graphlets are computed.
	                                  Default: -o kcore (degree, rand, natural/off, etc.).
	    --s2l                       : Direction of the ordering (default: largest to smallest).
	                                  Note to order from smallest to largest, set '--s2l'  
	-n, --neigh_ordering            : Strategy used to order the neighbors of each node. Default: kcore (degree, rand, natural, etc.)
	                                  Note only applicable if CSC/CSR is used (-r csc).
	    --s2l_neigh                 : Order direction for neighbor/csc ordering strategy
	                                  (e.g., --neigh_ordering degree --s2l_neigh, default: largest to smallest)
	---------------------------------------------------------------------------------
	-c, --counts,--macro            : Compute MACRO (GLOBAL) GRAPHLET FEATURES and save counts to file (e.g., --counts name.graphlets)
	-m, --micro                     : Compute MICRO (LOCAL) GRAPHLET FEATURES and save EDGE-by-MOTIF feature matrix (-m name.micro_graphlets)
	                                  Default: OFF. NOTE: Turn ON edge graphlet counting by specifying an output file, e.g., '-m name.micro_graphlets' 
	---------------------------------------------------------------------------------
	-v, --verbose                   : Output additional details to the screen. 
	-?, -h, --help                   : Print out this help menu. 


	REPRESENTATION: Example: ./pgd -r csc (adj, etc.)
	=================================================================================
	-r,   --rep                     : Graph representation [adj, csc, hybrid, auto, etc].
	                                  Default: Auto select optimal. 
		'adj'    - adjacency matrix   : dense n by n matrix, O(|V|^2) storage cost
		'csc'    - comp. sparse col   : large sparse graphs, O(|V|+|E|) storage cost
		'hybrid' -  csc + adj         : small sparse and dense graphs, O(|V|^2 + |V| + |E|) storage cost
	-l, --adj_limit                 : Threshold/limit for creating adj representation. Default: '-l 10000' (that is <10000 nodes).


	ORDERING TECHNIQUES: Example: ./pgd -o degree (kcore, rand, etc.)
	=================================================================================
	'degree',   'deg'                    : O(|V|)
	'kcore',                             : O(|E|)
	'rand', 'random'                     : O(|V|)
	'off',  'natural'                    

	 Other methods for ordering include: 
	'kcore_degree', 'kcore_deg'          : O(|V|)
	'degree_vol',   'deg_vol'            : O(|E|)
	'kcore_vol',                         : O(|E|)
	'deg_kcore_vol'                      : O(|E|)
	------------------------------------------------------------------
	NOTE: Default ordering is kcore (degeneracy order). For natural order, use '-o off' or '-o natural'




API and Sample Codes
------------------------

### Exact Sample Codes

* macro graphlet statistics
* micro graphlet statistics

Let us note that if the micro-level graphlet statistics are not needed, then it is recommended to use the macro graphlet decomposition algorithms. These are highly optimized for this task and thus are significantly more space-efficient while also faster and more scalable.


#### Macro graphlet statistics
Compute the global frequency of the various motif patterns with just two lines:

```cpp
// read graph, optimize alg/data structs, etc.
graphlet_core G("sample_graph.csv");
G.graphlet_decomposition();
```

Afterwards, it is easy to print or write global motif counts to a file.

```cpp
G.print_graphlet_counts(); // print to screen
```

or SAVE to a file,

```cpp
G.write_macro_stats(path);
```


#### Micro graphlet statistics
Compute the frequency of the various motif patterns with just two lines:

```cpp
// read graph, optimize alg/data structs, etc.
graphlet_core G("sample_graph.csv");
G.graphlet_decomposition_micro();
```

Afterwards, it is easy to print or write the motif counts to a file.

```cpp
G.print_micro_stats(); // print to screen
```

or SAVE to a file,

```cpp
G.write_micro_stats(path);
```


### GFD Sample Codes

#### Estimate GFD
To obtain a fast and accurate estimation of the graphlet frequency distribution, use the following:

```cpp
// Approximate GFD by sampling uniformly at random 10% of the edges (vertices) to use
G.graphlet_approximation(0.10);
```

Afterwards, the GFD can be approximated from these counts as follows:

```cpp
// Estimate graphlet distribution for connected and disconnected motifs
G.compute_GFD();
```


Distribution | API Call
------------------- | -------------
 Graphlet Freq. Distribution (GFD) | `compute_GFD()` 
 Connected GFD | `compute_connected_GFD()` 
 Disconnected GFD | `compute_disconnected_GFD()` 


	

#### Exact GFD
Exact graphlet distributions may also be computed fast by simply using an exact graphlet decomposition method from those expressed by the pgd library and then using the API calls above in the table.

```cpp
G.graphlet_decomposition();
G.compute_GFD();
```




Experimental Codes
-------------------

## Estimation Experiments


## Estimation Optimization Experiments


## Confidence Bounds

To compute the data needed for the confidence bound experiments, use the following:

	./pgd -f data/socfb-MIT.mtx --exp est_confidence --output results_confidence/

Note that if `-f data/socfb-MIT.mtx` is not set, then computes the data for all graphs in a certain directory. For instance,

	./pgd --exp est_confidence --output results_confidence/

To set the number of trials to repeat experiment, use the following:

	./pgd -f data/socfb-MIT.mtx --exp est_confidence --output results_confidence/ --trials 100
	
The above sets the number of trials to 100.


## Local/micro subgraph statistics

To output the  subgraph frequencies of each edge, use:

	./pgd -f data/test_case/socfb-MIT.mtx --exp micro_graphlet_estimation

This also outputs the frequency distribution for each subgraph pattern. By default, the results are saved in `dist`.
Additionally, the univariate statistics of each subgraph pattern are also stored in `dist`.



Documentation
--------------
The documentation is generated (using doxygen) by simply typing `make docs` in the root directory of pgd.

    make docs
    
This creates the `./docs` directory with the documentation.
To update the documentation, use `make cleandoc` then `make docs`.

Doxygen and graphvis are required and installed via homebrew (if not installed already).
Currently, this works only for Mac OSX and other Unix-based systems.



## Additional Info.

To generate the documentation you must have doxygen and graphviz installed. 
On Mac OSX these can be stalled using *homebrew* with the following commands:

    # install doxygen and graphviz using homebrew on Mac OSX
    brew install doxygen
    brew install graphviz

Afterwards, the documentation is generated by simply typing `make doc` in the root directory of pgd.
This creates the `./doc` directory with the documentation.

Note that in the unlikely case that *Doxyfile* does not exist, simply type `doxygen -g` to generate a default config file.
