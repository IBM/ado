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
```commandline
Identifier: datacontainer-532d8b6d
Basic Data:

  Label: person

  {'age': 2, 'name': 'mj'}


  Label: important_info

  ['t1',
   1,
   't2']

Tabular Data:

  Label: important_entities

      nodes          config      status provider  vcpu_size  cpu_family  wallClockRuntime
  0       5  A_f0.0-c1.0-n5          ok        A        1.0         0.0         84.453470
  1       3  A_f1.0-c1.0-n3          ok        A        1.0         1.0        151.585624
  2       3  A_f1.0-c1.0-n3          ok        A        1.0         1.0        155.028562
  3       3  A_f1.0-c0.0-n3          ok        A        0.0         1.0        206.744962
  4       4  A_f0.0-c0.0-n4          ok        A        0.0         0.0        145.129484
  5       3  A_f0.0-c1.0-n3          ok        A        1.0         0.0        168.365908
  6       5  A_f1.0-c1.0-n5          ok        A        1.0         1.0        105.637292
  7       5  A_f1.0-c0.0-n5          ok        A        0.0         1.0        135.910925
  8       4  A_f1.0-c1.0-n4          ok        A        1.0         1.0        116.314171
  9       2  A_f1.0-c0.0-n2          ok        A        0.0         1.0        378.316570
  10      5  A_f1.0-c0.0-n5          ok        A        0.0         1.0        117.941366
  11      5  A_f0.0-c0.0-n5          ok        A        0.0         0.0        106.070931
  12      4  A_f0.0-c1.0-n4          ok        A        1.0         0.0        106.670121
  13      3  A_f0.0-c1.0-n3          ok        A        1.0         0.0        170.156597
  14      2  A_f1.0-c1.0-n2          ok        A        1.0         1.0        291.904456
  15      5  A_f0.0-c1.0-n5          ok        A        1.0         0.0         86.230161
  16      2  A_f0.0-c0.0-n2          ok        A        0.0         0.0        335.208518
  17      3  A_f0.0-c0.0-n3          ok        A        0.0         0.0        221.510197
  18      4  A_f1.0-c0.0-n4          ok        A        0.0         1.0        158.706395
  19      2  A_f0.0-c1.0-n2          ok        A        1.0         0.0        272.997822
  20      5  A_f1.0-c1.0-n5          ok        A        1.0         1.0         96.847161
  21      5  A_f0.0-c0.0-n5          ok        A        0.0         0.0        130.305123
  22      3  A_f0.0-c0.0-n3          ok        A        0.0         0.0        216.394127
  23      3  A_f1.0-c0.0-n3          ok        A        0.0         1.0        236.171507
  24      3  B_f1.0-c0.0-n3          ok        B        0.0         1.0        220.198284
  25      4  B_f1.0-c0.0-n4          ok        B        0.0         1.0        202.482397
  26      5  B_f0.0-c0.0-n5          ok        B        0.0         0.0        103.905957
  27      4  B_f1.0-c0.0-n4          ok        B        0.0         1.0        193.559971
  28      2  B_f1.0-c1.0-n2          ok        B        1.0         1.0        298.819305
  29      4  B_f0.0-c0.0-n4          ok        B        0.0         0.0        113.876770
  30      3  B_f0.0-c0.0-n3          ok        B        0.0         0.0        153.516394
  31      3  B_f0.0-c0.0-n3          ok        B        0.0         0.0        184.448016
  32      5  B_f1.0-c0.0-n5          ok        B        0.0         1.0        141.990243
  33      2  B_f1.0-c0.0-n2          ok        B        0.0         1.0        346.070996
  34      5  B_f0.0-c0.0-n5          ok        B        0.0         0.0        112.705699
  35      2  B_f0.0-c1.0-n2          ok        B        1.0         0.0        184.935050
  36      4  B_f0.0-c0.0-n4          ok        B        0.0         0.0        132.541512
  37      5  B_f1.0-c0.0-n5          ok        B        0.0         1.0        168.791785
  38      2  B_f0.0-c0.0-n2          ok        B        0.0         0.0        225.179142
  39      3  B_f0.0-c0.0-n3          ok        B        0.0         0.0        176.288144
  40      2  B_f0.0-c0.0-n2          ok        B        0.0         0.0        228.143625
  41      2  B_f0.0-c1.0-n2          ok        B        1.0         0.0        166.748432
  42      5  B_f0.0-c0.0-n5          ok        B        0.0         0.0        113.885051
  43      3  B_f1.0-c0.0-n3          ok        B        0.0         1.0        273.712027
  44      2  C_f1.0-c1.0-n2          ok        C        1.0         1.0        363.285671
  45      3  C_f1.0-c0.0-n3  Timed out.        C        0.0         1.0        598.883466
  46      3  C_f1.0-c1.0-n3          ok        C        1.0         1.0        154.981347
  47      5  C_f0.0-c0.0-n5          ok        C        0.0         0.0        138.060516
  48      3  C_f0.0-c0.0-n3          ok        C        0.0         0.0        240.073585
  49      3  C_f0.0-c1.0-n3          ok        C        1.0         0.0        168.916364
  50      2  C_f0.0-c0.0-n2          ok        C        0.0         0.0        415.829285
  51      3  C_f0.0-c1.0-n3          ok        C        1.0         0.0        174.033562
  52      5  C_f0.0-c1.0-n5          ok        C        1.0         0.0         85.679467
  53      4  C_f0.0-c0.0-n4          ok        C        0.0         0.0        188.090878
  54      5  C_f1.0-c0.0-n5          ok        C        0.0         1.0        136.307105
  55      4  C_f1.0-c0.0-n4          ok        C        0.0         1.0        177.723598
  56      5  C_f1.0-c0.0-n5          ok        C        0.0         1.0        135.470500
  57      4  C_f1.0-c1.0-n4          ok        C        1.0         1.0        114.014369
  58      5  C_f0.0-c1.0-n5          ok        C        1.0         0.0         95.863261
  59      4  C_f0.0-c1.0-n4          ok        C        1.0         0.0        121.424925
  60      3  C_f1.0-c0.0-n3          ok        C        0.0         1.0        244.338875
  61      3  C_f1.0-c1.0-n3          ok        C        1.0         1.0        168.348592
  62      3  C_f0.0-c0.0-n3          ok        C        0.0         0.0        269.090664
  63      5  C_f0.0-c0.0-n5          ok        C        0.0         0.0        150.947150
  64      2  C_f1.0-c0.0-n2          ok        C        0.0         1.0        463.396539
  65      5  C_f1.0-c1.0-n5          ok        C        1.0         1.0         92.171414
  66      5  C_f1.0-c1.0-n5          ok        C        1.0         1.0        100.979775
  67      2  C_f0.0-c1.0-n2          ok        C        1.0         0.0        309.842324


Location Data:

  Label: entity_location

  mysql+pymysql://admin:somepass@localhost:3306/sql_sample_store_aaa123
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
