# -*- coding: utf-8 -*-
# @Time    : 2025/4/14 21:25
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: individual_confounders.py
# @Project : Causal3D-Net
import os, argparse
import logging
import radiomics
import numpy as np
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor


def extract_radiomics_features():
    parser = argparse.ArgumentParser(description="Extract handcrafted omics features")
    parser.add_argument("--input", type=str, default="/home/huangdn/Causal3D-Net/src/dataset/radiomics_read.xlsx", help="Input Excel file path.")
    parser.add_argument("--output", type=str, default="/home/huangdn/Causal3D-Net/src/data/radiomics_features.xlsx", help="Output Excel file path.")
    # parser.add_argument("--input", type=str, required=True, help="Input Excel file path.")
    # parser.add_argument("--output", type=str, required=True, help="Output Excel file path.")
    parser.add_argument("--params", type=str, default="/home/huangdn/Causal3D-Net/src/dataset/Params.yaml", help="Params file path.")
    parser.add_argument("--log_path", type=str, default="/home/huangdn/Causal3D-Net/src/logging_record/extract_radiomics_features.log", help="Logging file path.")
    args = parser.parse_args()

    rLogger = logging.getLogger('radiomics')
    handler = logging.FileHandler(filename=args.log_path, mode='w')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    rLogger.addHandler(handler)
    logger = rLogger.getChild('batch')
    radiomics.setVerbosity(logging.INFO)
    logger.info('pyradiomics version: %s', radiomics.__version__)
    logger.info('Loading Excel')
    try:
        flists = pd.read_excel(args.input).T
    except Exception:
        logger.error('Excel READ FAILED', exc_info=True)
        exit(-1)
    logger.info('Loading Done')
    logger.info('Patients: %d', len(flists.columns))

    if os.path.isfile(args.params):
        logger.info('Loading Params file')
        extractor = featureextractor.RadiomicsFeatureExtractor(args.params)
    else:  # Parameter file not found, use hardcoded settings instead
        logger.info('Loading Params base')
        settings = {}
        settings['binWidth'] = 25
        settings['resampledPixelSpacing'] = None
        settings['interpolator'] = sitk.sitkBSpline
        settings['enableCExtensions'] = True
        extractor = featureextractor.RadiomicsFeatureExtractor(**settings)
    logger.info('Enabled input images types: %s', extractor.enabledImagetypes)
    logger.info('Enabled features: %s', extractor.enabledFeatures)
    logger.info('Current settings: %s', extractor.settings)

    results = pd.DataFrame()
    for entry in flists:
        logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)",
                    entry + 1,
                    len(flists.T),  # 原来是 len(flists) 但是不对
                    flists[entry]['Image'],
                    flists[entry]['Mask'])
        imageFilepath = flists[entry]['Image']
        maskFilepath = flists[entry]['Mask']
        label = flists[entry].get('Label', None)

        if str(label).isdigit():
            label = int(label)
        else:
            label = None

        if (imageFilepath is not None) and (maskFilepath is not None):
            featureVector = flists[entry]  # This is a pandas Series
            featureVector['Image'] = os.path.basename(imageFilepath)
            featureVector['Mask'] = os.path.basename(maskFilepath)

        try:
            # PyRadiomics returns the result as an ordered dictionary, which can be easily converted to a pandas Series
            # The keys in the dictionary will be used as the index (labels for the rows), with the values of the features
            # as the values in the rows.
            result = pd.Series(extractor.execute(imageFilepath, maskFilepath, label))
            # featureVector = featureVector.append(result)  # 我的pandas版本过高，因此使用下面的更改方法
            featureVector = pd.concat([featureVector, result], ignore_index=False)

        except Exception:
            logger.error('FEATURE EXTRACTION FAILED:', exc_info=True)
        # To add the calculated features for this case to our data frame, the series must have a name (which will be the
        # name of the column.
        featureVector.name = entry
        # By specifying an 'outer' join, all calculated features are added to the data frame, including those not
        # calculated for previous cases. This also ensures we don't end up with an empty frame, as for the first patient
        # it is 'joined' with the empty data frame.
        results = results.join(featureVector, how='outer')  # If feature extraction failed, results will be all NaN
    logger.info('Extraction complete, writing Excel')
    results.T.to_excel(args.output, index=False, na_rep='NaN')
    logger.info('Excel writing complete')

    pass


if __name__ == '__main__':
    extract_radiomics_features()

