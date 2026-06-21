<!-- markdownlint-disable code-block-style -->
<!-- markdownlint-disable-next-line first-line-h1 -->
A `datacontainer` resource is one that contains data like tables, string and
locations. Its main purpose is to store output of `operations` that aren't
`samplestores` or `discoveryspaces`. For example, results of analyzing the
distribution of values in a space.

## creating a `datacontainer`

You currently can't create a `datacontainer` via the `ado` CLI. They are only
created as the result of applying certain operators.

## `datacontainer` contents

A `datacontainer` can contain the following types of data:

- lists, dicts, strings, numbers
- tabular data (DataFrames)
- location data (URLs)

A `datacontainer` resource has up to three top-level fields: `data`,
`locationData` and `tabularData`. Each of these is a dictionary whose values are
data objects and keys are the names of the data. The `tabularData` field
contains items that are DataFrames. The `locationData` field contains items that
are URLs. The `data` field contains items that are JSON serializable types:
lists, dicts, string and numbers. Note, in the `data` field all data in
containers must also be lists, dicts, strings or numbers.

## Accessing the contents of a `datacontainer`

### via `ado` cli

The data in a `datacontainer` is stored directly in the resource description.
Hence `ado get datacontainer $ID` will output it. However, depending on what is
stored this may not be the best way to view it. Instead, you can try
`ado describe datacontainer` which will format the contents e.g.

<!-- markdownlint-disable line-length -->
```terminaloutput
Identifier: datacontainer-219160f1
                                                                                                           
 ────────────────────────────────────────────── Basic Data ─────────────────────────────────────────────── 
                                                                                                           
    Label: 'person'                                                                                        
    {'name': 'mj', 'age': 2}                                                                               
                                                                                                           
                                                                                                           
    Label: 'important_info'                                                                                
    ['t1', 1, 't2']                                                                                        
                                                                                                           
 ───────────────────────────────────────────────────────────────────────────────────────────────────────── 
 ───────────────────────────────────────────── Tabular Data ────────────────────────────────────────────── 
                                                                                                           
    Label: 'important_entities'                                                                            
                                                                                                           
     config            cpu_family    vcpu_size    nodes    wallClockRuntime      status       provider     
    ───────────────────────────────────────────────────────────────────────────────────────────────────    
     A_f0.0-c1.0-n5    0.0           1.0          5        84.45346999168396     ok           A            
     A_f1.0-c1.0-n3    1.0           1.0          3        151.58562421798706    ok           A            
     A_f1.0-c1.0-n3    1.0           1.0          3        155.02856159210205    ok           A            
     A_f1.0-c0.0-n3    1.0           0.0          3        206.74496150016785    ok           A            
     A_f0.0-c0.0-n4    0.0           0.0          4        145.12948369979858    ok           A            
     A_f0.0-c1.0-n3    0.0           1.0          3        168.36590766906738    ok           A            
     A_f1.0-c1.0-n5    1.0           1.0          5        105.63729166984558    ok           A            
     A_f1.0-c0.0-n5    1.0           0.0          5        135.91092538833618    ok           A            
     A_f1.0-c1.0-n4    1.0           1.0          4        116.31417059898376    ok           A            
     A_f1.0-c0.0-n2    1.0           0.0          2        378.31657004356384    ok           A            
     A_f1.0-c0.0-n5    1.0           0.0          5        117.94136571884157    ok           A            
     A_f0.0-c0.0-n5    0.0           0.0          5        106.0709307193756     ok           A            
     A_f0.0-c1.0-n4    0.0           1.0          4        106.67012143135072    ok           A            
     A_f0.0-c1.0-n3    0.0           1.0          3        170.15659737586975    ok           A            
     A_f1.0-c1.0-n2    1.0           1.0          2        291.90445613861084    ok           A            
     A_f0.0-c1.0-n5    0.0           1.0          5        86.23016095161438     ok           A            
     A_f0.0-c0.0-n2    0.0           0.0          2        335.2085180282593     ok           A            
     A_f0.0-c0.0-n3    0.0           0.0          3        221.5101969242096     ok           A            
     A_f1.0-c0.0-n4    1.0           0.0          4        158.70639538764954    ok           A            
     A_f0.0-c1.0-n2    0.0           1.0          2        272.99782156944275    ok           A            
     A_f1.0-c1.0-n5    1.0           1.0          5        96.8471610546112      ok           A            
     A_f0.0-c0.0-n5    0.0           0.0          5        130.30512285232544    ok           A            
     A_f0.0-c0.0-n3    0.0           0.0          3        216.394127368927      ok           A            
     A_f1.0-c0.0-n3    1.0           0.0          3        236.1715066432953     ok           A            
     B_f1.0-c0.0-n3    1.0           0.0          3        220.19828414916992    ok           B            
     B_f1.0-c0.0-n4    1.0           0.0          4        202.48239731788635    ok           B            
     B_f0.0-c0.0-n5    0.0           0.0          5        103.90595746040344    ok           B            
     B_f1.0-c0.0-n4    1.0           0.0          4        193.55997109413147    ok           B            
     B_f1.0-c1.0-n2    1.0           1.0          2        298.8193049430847     ok           B            
     B_f0.0-c0.0-n4    0.0           0.0          4        113.87676978111269    ok           B            
     B_f0.0-c0.0-n3    0.0           0.0          3        153.51639366149902    ok           B            
     B_f0.0-c0.0-n3    0.0           0.0          3        184.44801592826843    ok           B            
     B_f1.0-c0.0-n5    1.0           0.0          5        141.99024295806885    ok           B            
     B_f1.0-c0.0-n2    1.0           0.0          2        346.0709958076477     ok           B            
     B_f0.0-c0.0-n5    0.0           0.0          5        112.7056987285614     ok           B            
     B_f0.0-c1.0-n2    0.0           1.0          2        184.935049533844      ok           B            
     B_f0.0-c0.0-n4    0.0           0.0          4        132.5415120124817     ok           B            
     B_f1.0-c0.0-n5    1.0           0.0          5        168.79178500175476    ok           B            
     B_f0.0-c0.0-n2    0.0           0.0          2        225.1791422367096     ok           B            
     B_f0.0-c0.0-n3    0.0           0.0          3        176.28814435005188    ok           B            
     B_f0.0-c0.0-n2    0.0           0.0          2        228.14362454414368    ok           B            
     B_f0.0-c1.0-n2    0.0           1.0          2        166.74843192100525    ok           B            
     B_f0.0-c0.0-n5    0.0           0.0          5        113.88505148887634    ok           B            
     B_f1.0-c0.0-n3    1.0           0.0          3        273.7120273113251     ok           B            
     C_f1.0-c1.0-n2    1.0           1.0          2        363.2856709957123     ok           C            
     C_f1.0-c0.0-n3    1.0           0.0          3        598.8834657669067     Timed out.   C            
     C_f1.0-c1.0-n3    1.0           1.0          3        154.9813470840454     ok           C            
     C_f0.0-c0.0-n5    0.0           0.0          5        138.0605161190033     ok           C            
     C_f0.0-c0.0-n3    0.0           0.0          3        240.07358503341675    ok           C            
     C_f0.0-c1.0-n3    0.0           1.0          3        168.9163637161255     ok           C            
     C_f0.0-c0.0-n2    0.0           0.0          2        415.8292849063873     ok           C            
     C_f0.0-c1.0-n3    0.0           1.0          3        174.0335624217987     ok           C            
     C_f0.0-c1.0-n5    0.0           1.0          5        85.67946743965149     ok           C            
     C_f0.0-c0.0-n4    0.0           0.0          4        188.09087824821472    ok           C            
     C_f1.0-c0.0-n5    1.0           0.0          5        136.3071050643921     ok           C            
     C_f1.0-c0.0-n4    1.0           0.0          4        177.72359776496887    ok           C            
     C_f1.0-c0.0-n5    1.0           0.0          5        135.47050046920776    ok           C            
     C_f1.0-c1.0-n4    1.0           1.0          4        114.01436853408812    ok           C            
     C_f0.0-c1.0-n5    0.0           1.0          5        95.86326050758362     ok           C            
     C_f0.0-c1.0-n4    0.0           1.0          4        121.42492485046388    ok           C            
     C_f1.0-c0.0-n3    1.0           0.0          3        244.33887457847595    ok           C            
     C_f1.0-c1.0-n3    1.0           1.0          3        168.34859228134155    ok           C            
     C_f0.0-c0.0-n3    0.0           0.0          3        269.0906641483307     ok           C            
     C_f0.0-c0.0-n5    0.0           0.0          5        150.9471504688263     ok           C            
     C_f1.0-c0.0-n2    1.0           0.0          2        463.396538734436      ok           C            
     C_f1.0-c1.0-n5    1.0           1.0          5        92.17141437530518     ok           C            
     C_f1.0-c1.0-n5    1.0           1.0          5        100.97977471351624    ok           C            
     C_f0.0-c1.0-n2    0.0           1.0          2        309.8423240184784     ok           C            
                                                                                                           
 ───────────────────────────────────────────────────────────────────────────────────────────────────────── 
 ───────────────────────────────────────────── Location Data ───────────────────────────────────────────── 
                                                                                                           
    Label: 'entity_location'                                                                               
    mysql+pymysql://admin:somepass@localhost:3306/sql_sample_store_aaa123                                  
                                                                                                           
 ───────────────────────────────────────────────────────────────────────────────────────────────────────── 
```
<!-- markdownlint-enable line-length -->

### programmatically

For certain data, like large tables, it may be more convenient to access the
data programmatically.

If you do `ado get datacontainer $RESOURCEID -o yaml > data.yaml`. Then the
following snippet shows how to access the data in python

```python

from orchestrator.core.datacontainer.resource import DataContainer
import yaml

with open('data.yaml') as f:
    d = DataContainer.model_validate(yaml.safe_load(f))

# for tabular data
for table in d.tabularData.values():
    # Get the table as pandas dataframe
    df = table.dataframe()
    ...

for location in d.locationData.values():
    # Each value in the locationData is a subclass of orchestrator.utilities.location.ResourceLocation
    print(location.url().unicode_string())
```
