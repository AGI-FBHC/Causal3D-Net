# -*- coding: utf-8 -*-
# @Time    : 2025/4/14 21:25
# @Author  : D.N. Huang
# @Email   : CarlCypress@yeah.net
# @FileName: extract_radiomics.py
# @Project : Causal3D-Net
import logging
import radiomics
import numpy as np
import os, argparse
import pandas as pd
import SimpleITK as sitk
from radiomics import featureextractor
from concurrent.futures import ProcessPoolExecutor


def process_patient(entry, flists, extractor, logger):
    imageFilepath = flists[entry]['Image']
    maskFilepath = flists[entry]['Mask']
    label = flists[entry].get('Label', None)
    logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)",
                entry + 1,
                len(flists.T),  # 原来是 len(flists) 但是不对
                imageFilepath,
                maskFilepath)

    if str(label).isdigit():
        label = int(label)
    else:
        label = None

    featureVector = flists[entry]  # This is a pandas Series
    featureVector['Image'] = os.path.basename(imageFilepath)
    featureVector['Mask'] = os.path.basename(maskFilepath)

    try:
        # PyRadiomics returns the result as an ordered dictionary, which can be easily converted to a pandas Series
        result = pd.Series(extractor.execute(imageFilepath, maskFilepath, label))
        featureVector = pd.concat([featureVector, result], ignore_index=False)
    except Exception:
        logger.error('FEATURE EXTRACTION FAILED for patient: %s', entry, exc_info=True)

    featureVector.name = entry
    return featureVector


def extract_radiomics_features(excel_input_path, output, params, log_path, process_num):
    rLogger = logging.getLogger('radiomics')
    handler = logging.FileHandler(filename=log_path, mode='w')
    handler.setFormatter(logging.Formatter('%(levelname)s:%(name)s: %(message)s'))
    rLogger.addHandler(handler)
    logger = rLogger.getChild('batch')
    radiomics.setVerbosity(logging.INFO)
    logger.info('pyradiomics version: %s', radiomics.__version__)
    logger.info('Loading Excel')
    try:
        flists = pd.read_excel(excel_input_path).T
        # Check Table
        for entry in flists:
            if not os.path.isfile(flists[entry]["Image"]) or not os.path.isfile(flists[entry]["Mask"]):
                logger.info(f"{flists[entry]['Image']} or {flists[entry]['Mask']} dose not exist")
                logger.error("There is an issue with the file path", exc_info=True)
                exit(-1)
    except Exception:
        logger.error('Excel READ FAILED', exc_info=True)
        exit(-1)
    logger.info('Loading Done')
    logger.info('Patients: %d', len(flists.columns))

    if os.path.isfile(params):
        logger.info('Loading Params file')
        extractor = featureextractor.RadiomicsFeatureExtractor(params)
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

    # Initialize the ProcessPoolExecutor with number of workers as desired
    with ProcessPoolExecutor(max_workers=process_num) as executor:
        # Submit each patient task to the executor
        futures = {executor.submit(process_patient, entry, flists, extractor, logger): entry
                   for entry in flists.columns}

        for future in futures:
            entry = futures[future]
            try:
                featureVector = future.result()
                # Join the featureVector to the results DataFrame
                results = results.join(featureVector,
                                       how='outer')  # If feature extraction failed, results will be all NaN
            except Exception as e:
                logger.error('Error processing patient %s: %s', entry, e)

    # for entry in flists:
    #     logger.info("(%d/%d) Processing Patient (Image: %s, Mask: %s)",
    #                 entry + 1,
    #                 len(flists.T),  # 原来是 len(flists) 但是不对
    #                 flists[entry]['Image'],
    #                 flists[entry]['Mask'])
    #     imageFilepath = flists[entry]['Image']
    #     maskFilepath = flists[entry]['Mask']
    #     label = flists[entry].get('Label', None)
    #
    #     if str(label).isdigit():
    #         label = int(label)
    #     else:
    #         label = None
    #
    #     if (imageFilepath is not None) and (maskFilepath is not None):
    #         featureVector = flists[entry]  # This is a pandas Series
    #         featureVector['Image'] = os.path.basename(imageFilepath)
    #         featureVector['Mask'] = os.path.basename(maskFilepath)
    #
    #     try:
    #         # PyRadiomics returns the result as an ordered dictionary, which can be easily converted to a pandas Series
    #         # The keys in the dictionary will be used as the index (labels for the rows), with the values of the features
    #         # as the values in the rows.
    #         result = pd.Series(extractor.execute(imageFilepath, maskFilepath, label))
    #         # featureVector = featureVector.append(result)  # 我的pandas版本过高，因此使用下面的更改方法
    #         featureVector = pd.concat([featureVector, result], ignore_index=False)
    #
    #     except Exception:
    #         logger.error('FEATURE EXTRACTION FAILED:', exc_info=True)
    #     # To add the calculated features for this case to our data frame, the series must have a name (which will be the
    #     # name of the column.
    #     featureVector.name = entry
    #     # By specifying an 'outer' join, all calculated features are added to the data frame, including those not
    #     # calculated for previous cases. This also ensures we don't end up with an empty frame, as for the first patient
    #     # it is 'joined' with the empty data frame.
    #     results = results.join(featureVector, how='outer')  # If feature extraction failed, results will be all NaN
    logger.info('Extraction complete, writing CSV')
    results.T.to_csv(output, index=False, na_rep='NaN')
    logger.info('CSV writing complete')
    return results


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Extract handcrafted omics features")
    parser.add_argument(
        "--input",
        type=str,
        default="/home/huangdn/Causal3D-Net/src/dataset/dataset_for_radiomics_read.xlsx",
        # required=True,
        help="Input Excel file path."
    )
    parser.add_argument(
        "--output",
        type=str,
        default="/home/huangdn/Causal3D-Net/src/dataset/radiomics_features.csv",
        # required=True,
        help="Output Excel file path."
    )
    parser.add_argument(
        "--params",
        type=str,
        default="/home/huangdn/Causal3D-Net/src/config/Params.yaml",
        help="Params file path."
    )
    parser.add_argument(
        "--log_path",
        type=str,
        default="/home/huangdn/Causal3D-Net/src/logging_record/extract_radiomics_features.log",
        help="Logging file path."
    )
    parser.add_argument(
        "--process_num",
        type=int,
        default=8,
        help="Number of concurrent processes to run, be careful not to exceed the number of CPU cores."
    )
    args = parser.parse_args()
    extract_radiomics_features(
        args.input,
        args.output,
        args.params,
        args.log_path,
        args.process_num,
    )

