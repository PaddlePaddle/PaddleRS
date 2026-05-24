[简体中文](indices_cn.md) | English

# Remote Sensing Indices

Through `paddlers.transforms.AppendIndex` operator remote sensing index can be calculated and appended to the input image in the last band. When you build the `AppendIndex` object, you need to pass in the remote sensing index name and a dictionary containing the band-index correspondence (the key in the dictionary is the band name and the index number counts from 1).

## List of PaddleRS Supported Remote Sensing Indices

|Remote sensing index name|Full name|Purpose|Reference|
|-----------|----|---|--------|
| `'ARI'` | Anthocyanin Reflectance Index | vegetation | https://doi.org/10.1562/0031-8655(2001)074%3C0038:OPANEO%3E2.0.CO;2 |
| `'ARI2'` | Anthocyanin Reflectance Index 2 | vegetation | https://doi.org/10.1562/0031-8655(2001)074%3C0038:OPANEO%3E2.0.CO;2 |
| `'ARVI'` | Atmospherically Resistant Vegetation Index | vegetation | https://doi.org/10.1109/36.134076 |
| `'AWEInsh'` | Automated Water Extraction Index | Water | https://doi.org/10.1016/j.rse.2013.08.029 |
| `'AWEIsh'` | Automated Water Extraction Index with Shadows Elimination | Water | https://doi.org/10.1016/j.rse.2013.08.029 |
| `'BAI'` | Burned Area Index | burned area | https://digital.csic.es/bitstream/10261/6426/1/Martin_Isabel_Serie_Geografica.pdf |
| `'BI'` | Bare Soil Index | cities and towns | http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.465.8749&rep=rep1&type=pdf |
| `'BLFEI'` | Built-Up Land Features Extraction Index | cities and towns | https://doi.org/10.1080/10106049.2018.1497094 |
| `'BNDVI'` | Blue Normalized Difference Vegetation Index | vegetation | https://doi.org/10.1016/S1672-6308(07)60027-4 |
| `'BWDRVI'` | Blue Wide Dynamic Range Vegetation Index | vegetation | https://doi.org/10.2135/cropsci2007.01.0031 |
| `'BaI'` | Bareness Index | cities and towns | https://doi.org/10.1109/IGARSS.2005.1525743 |
| `'CIG'` | Chlorophyll Index Green | vegetation | https://doi.org/10.1078/0176-1617-00887 |
| `'CSI'` | Char Soil Index | burned area | https://doi.org/10.1016/j.rse.2005.04.014 |
| `'CSIT'` | Char Soil Index Thermal | burned area | https://doi.org/10.1080/01431160600954704 |
| `'DBI'` | Dry Built-Up Index | cities and towns | https://doi.org/10.3390/land7030081 |
| `'DBSI'` | Dry Bareness Index | cities and towns | https://doi.org/10.3390/land7030081 |
| `'DVI'` | Difference Vegetation Index | vegetation | https://doi.org/10.1016/0034-4257(94)00114-3 |
| `'EBBI'` | Enhanced Built-Up and Bareness Index | cities and towns | https://doi.org/10.3390/rs4102957 |
| `'EVI'` | Enhanced Vegetation Index | vegetation | https://doi.org/10.1016/S0034-4257(96)00112-5 |
| `'EVI2'` | Two-Band Enhanced Vegetation Index | vegetation | https://doi.org/10.1016/j.rse.2008.06.006 |
| `'FCVI'` | Fluorescence Correction Vegetation Index | vegetation | https://doi.org/10.1016/j.rse.2020.111676 |
| `'GARI'` | Green Atmospherically Resistant Vegetation Index | vegetation | https://doi.org/10.1016/S0034-4257(96)00072-7 |
| `'GBNDVI'` | Green-Blue Normalized Difference Vegetation Index | vegetation | https://doi.org/10.1016/S1672-6308(07)60027-4 |
| `'GLI'` | Green Leaf Index | vegetation | http://dx.doi.org/10.1080/10106040108542184 |
| `'GRVI'` | Green Ratio Vegetation Index | vegetation | https://doi.org/10.2134/agronj2004.0314 |
| `'IPVI'` | Infrared Percentage Vegetation Index | vegetation | https://doi.org/10.1016/0034-4257(90)90085-Z |
| `'LSWI'` | Land Surface Water Index | Water | https://doi.org/10.1016/j.rse.2003.11.008 |
| `'MBI'` | Modified Bare Soil Index | cities and towns | https://doi.org/10.3390/land10030231 |
| `'MGRVI'` | Modified Green Red Vegetation Index | vegetation | https://doi.org/10.1016/j.jag.2015.02.012 |
| `'MNDVI'` | Modified Normalized Difference Vegetation Index | vegetation | https://doi.org/10.1080/014311697216810 |
| `'MNDWI'` | Modified Normalized Difference Water Index | Water | https://doi.org/10.1080/01431160600589179 |
| `'MNLI'` | Modified Non-Linear Vegetation Index | vegetation | https://doi.org/10.1109/TGRS.2003.812910 |
| `'MSI'` | Moisture Stress Index | vegetation | https://doi.org/10.1016/0034-4257(89)90046-1 |
| `'NBLI'` | Normalized Difference Bare Land Index | cities and towns | https://doi.org/10.3390/rs9030249 |
| `'NDSI'` | Normalized Difference Snow Index | Snow | https://doi.org/10.1109/IGARSS.1994.399618 |
| `'NDVI'` | Normalized Difference Vegetation Index | vegetation | https://ntrs.nasa.gov/citations/19740022614 |
| `'NDWI'` | Normalized Difference Water Index | Water | https://doi.org/10.1080/01431169608948714 |
| `'NDYI'` | Normalized Difference Yellowness Index | vegetation | https://doi.org/10.1016/j.rse.2016.06.016 |
| `'NIRv'` | Near-Infrared Reflectance of Vegetation | vegetation | https://doi.org/10.1126/sciadv.1602244 |
| `'PSRI'` | Plant Senescing Reflectance Index | vegetation | https://doi.org/10.1034/j.1399-3054.1999.106119.x |
| `'RI'` | Redness Index | vegetation | https://www.documentation.ird.fr/hor/fdi:34390 |
| `'SAVI'` | Soil-Adjusted Vegetation Index | vegetation | https://doi.org/10.1016/0034-4257(88)90106-X |
| `'SWI'` | Snow Water Index | Snow | https://doi.org/10.3390/rs11232774 |
| `'TDVI'` | Transformed Difference Vegetation Index | vegetation | https://doi.org/10.1109/IGARSS.2002.1026867 |
| `'UI'` | Urban Index | cities and towns | https://www.isprs.org/proceedings/XXXI/congress/part7/321_XXXI-part7.pdf |
| `'VIG'` | Vegetation Index Green | vegetation | https://doi.org/10.1016/S0034-4257(01)00289-9 |
| `'WI1'` | Water Index 1 | Water | https://doi.org/10.3390/rs11182186 |
| `'WI2'` | Water Index 2 | Water | https://doi.org/10.3390/rs11182186 |
| `'WRI'` | Water Ratio Index | Water | https://doi.org/10.1109/GEOINFORMATICS.2010.5567762 |

## Band Name and Description

|    Band name    |     Description    | Reference wavelength range(μm) |  \*Reference wavelength source  |
|---------------|------------------|------------------|------------------|
|     `'b'`     | Blue             | *0.450 - 0.515*  | *Landsat8*       |
|     `'g'`     | Green            | *0.525 - 0.600*  | *Landsat8*       |
|     `'r'`     | Red              | *0.630 - 0.680*  | *Landsat8*       |
|    `'re1'`    | Red Edge 1       | *0.698 - 0.713*  | *Sentinel2*      |
|    `'re2'`    | Red Edge 2       | *0.733 - 0.748*  | *Sentinel2*      |
|    `'re3'`    | Red Edge 3       | *0.773 - 0.793*  | *Sentinel2*      |
|     `'n'`     | NIR              | *0.845 - 0.885*  | *Landsat8*       |
|    `'s1'`     | SWIR 1           | *1.560 - 1.660*  | *Landsat8*       |
|    `'s2'`     | SWIR 2           | *2.100 - 2.300*  | *Landsat8*       |
|    `'t'`      | Thermal Infrared | *10.40 - 12.50*  | *Landsat7*       |
|    `'t1'`     | Thermal 1        | *10.60 - 11.19*  | *Landsat8*       |
|    `'t2'`     | Thermal 2        | *11.50 - 12.51*  | *Landsat8*       |
