import os
import pandas as pd
import numpy as np
import argparse

# Specify data location
CLINICAL_Patient_File = 'data_clinical_supp_patient.txt' # patient data. 
CLINICAL_Sample_File = 'data_clinical_supp_sample.txt' # patient's breast tumor data
CNA_File = 'data_CNA.txt' # copy number alteration data. See meta_CNA.txt for more information
EXP_File = 'data_expression.txt' # gene expression data. See meta_expression.txt for more information
TARGET_GENES = 'TP53	FOXO3	NCOR1	PIK3CA	SETD2	BIRC6	TG	GATA3	ARID2	NCOR2	CBFB	BAP1	STAB2	MUC16	FOXP1	RYR2	FANCD2	KMT2C	CDH1	NF1	USH2A	MTAP	ERBB3	MAP3K1	SF3B1	MLL2	RB1	COL6A3	UTRN	PTEN	BRCA2	CASP8	AHNAK	ALK	KDM6A	AGMO	SYNE1	ARID1A	AKT1	LIPI	ASXL2	TAF1	APC	SETD1A	AKAP9	UBR5	LAMA2	MAP2K4	BRIP1	PRKCE	PIK3R1	HERC2	FBXW7	AHNAK2	GPS2	THSD7A	MYH9	ZFP36L1	GPR32	GH1	L1CAM	SMAD4	NOTCH1	JAK1	DNAH2	COL22A1	TBX3	COL12A1	DNAH5	CTCF	KRAS	CACNA2D3	TTYH1	ERBB4	MBL2	SIK1	AKT2	ARID5B	THADA	FRMD3	ATR	RUNX1	BRCA1	EGFR	PRKCQ	LIFR	SMARCC2	MEN1	ROS1	LAMB3	USP9X	RPGR	AFF2	CDKN1B	PRPS2	PALLD	SHANK2	PTPRM	PTPRD	ASXL1	GPR124	CHEK2	ERBB2	CDKN2A	MLLT4	PDE4DIP	SMARCC1	CTNNA3	MAP3K10	LARGE	SETDB1	ARID1B	DCAF4L2	NCOA3	DNAH11	MAP3K13	BCAS3	PBRM1	NRG3	HDAC9	ACVRL1	NDFIP1	USP28	CHD1	OR6A2	CLK3	PRKCZ	MYO3A	PRKG1	FLT3	BRAF	PRR16	PRKACG	FAM20C	KDM3A	NPNT	NEK1	NF2	FANCA	MYO1A	PPP2R2A	STK11	EP300	CTNNA1	FOXO1	SGCD	SBNO1	HIST1H2BC	SPACA1	SIK2	DTWD2	GLDC	NR2F1	MAGEA8	KLRG1	TAF4B	HRAS	RASGEF1B	SMAD2	NR3C1	LDLRAP1	NT5E	PTPN22	CLRN2	CCND3	SMARCB1	TBL1XR1	PPP2CB	SIAH1	SMARCD1	STMN2	NRAS	AGTR2'

# Read files
def parse_clinical_files(data_folder):
    # parse patient data
    patient_data = []
    patient_rows = 0
    with open(os.path.join(data_folder, CLINICAL_Patient_File), 'r') as f:
        for line in f:
            line = line.strip('\n')
            line = line.split('\t')
            patient_rows += 1
            if patient_rows == 1:
                patient_headers = line
            else:
                patient_data.append(line)
  
    # Create a data frame to store patient data
    patient_data = pd.DataFrame(patient_data, columns = patient_headers)
    #print('patients: ', patient_rows, len(patient_headers))
    # parse tumor sample data
    sample_data = []
    sample_rows = 0
    
    with open(os.path.join(data_folder, CLINICAL_Sample_File), 'r') as f:
        for line in f:
            line = line.strip('\n')
            line = line.split('\t')
            sample_rows += 1
            if sample_rows == 1:
                sample_headers = line
            else:
                sample_data.append(line)
    # Create a data frame to store sample data
    sample_data = pd.DataFrame(sample_data, columns = sample_headers)
    #print('sample: ', sample_rows, len(sample_headers))
    
    return patient_data, sample_data

def parse_geneExpression(data_folder):
    # parsing gene expression data
    gene_list = TARGET_GENES.split('\t')
    exp_data = []
    rows = 0
    with open(os.path.join(data_folder, EXP_File), 'r') as f:
        for line in f:
            line = line.strip('\n')
            line = line.split('\t')
            rows = rows + 1
            if rows == 1:
                headers = line
            else:
                if line[0] in gene_list or line[1] in gene_list:
                    exp_data.append(line)
                else:
                    continue
    exp_data = pd.DataFrame(exp_data, columns = headers)
    #print('GE: ', rows, len(headers))
    return exp_data

def parse_CNA(data_folder):
    gene_list = TARGET_GENES.split('\t')
    rows = 0
    cnv_data = []
    with open(os.path.join(data_folder, CNA_File), 'r') as f:
        for line in f:
            line = line.strip('\n')
            line = line.split('\t')
            rows += 1
            if rows == 1:
               headers = line
            else:
                if line[0] in gene_list or line[1] in gene_list:
                    cnv_data.append(line)
                else:
                    continue
    cnv_data = pd.DataFrame(cnv_data, columns = headers)
    #print('cna', rows, len(headers))
    return cnv_data    

def profile_data(df):
    # report size (rows, columns), features, features' values
    shape = df.shape
    features = list(df)
    features_dict = {}
    missing_values = ['NA', 'null', np.NaN]
    troubled_columns = []
    for feature in features:
        flag = False
        values = df[feature]
        #if values.isnull().values.any():
        #    flag = True
        values = list(set(values))
        for value in missing_values:
            if value in values: 
                values.remove(value)
                flag = True
        if flag == True:
            troubled_columns.append(feature)
        features_dict[feature] = values
    
    return shape, features_dict, troubled_columns



def remove_missingValues(df):
    # remove row that have missing values
    troubled_rows = []
    for index, row in df.iterrows():
        flag = False
        for feature in list(df):
            if row[feature] in ['NA', 'null'] or pd.isnull(row[feature]):
                flag = True
                break
        if flag == True:
            troubled_rows.append(index)
        else:
            continue
    new_df = df.copy()
    new_df = new_df.drop(troubled_rows)
    
    return new_df

def clean_data(df):
    # remove rows that have missing values from data
    shape, features_dict, troubled_columns = profile_data(df)
    troubled_rows = []
    for index, row in df.iterrows():
        flag = False
        for feature in troubled_columns:
            if row[feature] in ['NA', 'null', np.NaN]:  
                flag = True
                break
        if flag == True:
            troubled_rows.append(index)
    new_df = df.copy()
    new_df = new_df.drop(troubled_rows)
    return new_df, troubled_rows

def print_stats(filename, df):
    shape, features_dict, troubled_columns = profile_data(df)
    print('This df has {} rows and {} columns'.format(shape[0], shape[1]), file = filename)
    print('This df has the following features: ', file = filename)
    for feature in features_dict.keys():
        print(feature, ': ', features_dict[feature], file = filename)
    print('The following features have missing values: {}'.format(troubled_columns),  file = filename)

def generate_Features(df, excluded_features):
    # classify features into categorical, numerical and binary
    categorical_features, numerical_features, binary_features = classify_features(df, excluded_features)
    new_df = df.copy()
    # convert values in binary_feature to 0 and 1
    for feature in binary_features:
        binary_data = toBinary(df[feature])
        new_df[feature] = binary_data
    # convert string values in numerical feature to floats
    for feature in numerical_features:
        numerical_data = [float(x) for x in df[feature]]
        new_df[feature] = numerical_data
    # convert categorical values in categorical features to binary using one-hot encoding    
    new_df = new_df.drop(columns = categorical_features)
    for feature in categorical_features:
        mat = one_hot_encoding(df[feature])
        columns = [] # new columns replaced original dropped column
        for key in list(set(df[feature])):
            name = feature + '_' + str(key)
            columns.append(name)
        temp_df = pd.DataFrame(mat, columns = columns)
        # reset the index
        temp_df.reset_index(drop = True, inplace = True)
        new_df.reset_index(drop = True, inplace = True)
        new_df = pd.concat([new_df, temp_df], axis = 1)
      
    # convert the class feature to categorical values
    if 'OS_MONTHS' in numerical_features:
        class_data = new_df['OS_MONTHS']
        class_data = np.divide(class_data, 12)
        new_class_data = []
        for row in class_data:
            if row < 5:
                new_class_data.append(0) # death at 5 years
            elif row < 10:
                new_class_data.append(1) # alive at 5 years
            else:
                new_class_data.append(2) # alive at 10 years
        new_df['OS_MONTHS'] = new_class_data
    #print(numerical_features)
    #print(categorical_features)
    #print(binary_features)
    return new_df

def toBinary(data):
    # convert binary features' values to 1 and 0
    keys = list(set(data))
    binary_data = []
    for row in data:
        if row == keys[0]:
            binary_data.append(0)
        elif row == keys[1]:
            binary_data.append(1)
        else:
            binary_data.append(2)
  
    return binary_data

def one_hot_encoding(data):
    keys = list(set(data))
    mat = np.zeros((len(data), len(keys)))
    row_index = 0
    for row in data:
        if row in keys:
            col_index = keys.index(row)
            mat[row_index, col_index] = 1
            row_index += 1
    
    return mat


def isFloat(list_values):
    # determine if all values in a list can be converted to float
    flag = True
    for value in list_values:
        try:
            float(value)
        except:
            flag = False
        if flag == False:
            break
        else:
            continue
    
    return flag

def classify_features(df, excluded_features):
    # classify features to numerical, categorical and binary features
    shape, features_dict, troubled_columns = profile_data(df)
    categorical_features = []
    numerical_features = []
    binary_features = []
    features = list(features_dict.keys())
    # exclude feature in excluded_features
    for feature in excluded_features:
        features.remove(feature)
    for feature in features:
        values = set(features_dict[feature])
        #print(feature, ": ", len(values))
        if len(values) == 2:
            binary_features.append(feature)
        elif isFloat(values) == True and len(values) > 20:
                numerical_features.append(feature)
        else:
            categorical_features.append(feature)
                
    return categorical_features, numerical_features, binary_features


def transpose(df):
    # transpose df
    headers = list(df)
    df = df.drop(columns = headers[1]) # remove ENTREZ ID
    data = df.values
    new_headers = data[:, 0]
    new_headers = np.insert(new_headers,0,'PATIENT_ID')
    data = data.T
    data = data[1:, :]
    patient_id = headers[2:]
    data = np.insert(data, 0,patient_id, 1)
    new_df = pd.DataFrame(data, columns = new_headers)    
    return new_df    

def main(args):
    output_folder = args.Output
    data_folder = args.Data
    outfile = args.out
    if not os.path.exists(os.path.join(output_folder,'clinical_data.csv')):
        patient_data, sample_data = parse_clinical_files(data_folder)
        # merge patient and sample data by patient's ID
        df1 = pd.merge(patient_data, sample_data, on = 'PATIENT_ID')
        df1 = remove_missingValues(df1)
        excluded_attributes = ['PATIENT_ID', 'SAMPLE_ID', 'OS_STATUS', 'VITAL_STATUS']
        df2 =  generate_Features(df1, excluded_attributes)
        df2.to_csv(os.path.join(output_folder, 'clinical_data.csv'), index = False)
        #print('clinical: ', df2.shape)
    if not os.path.exists(os.path.join(output_folder, 'reducedGE.csv')):
        # parse gene expression data
        exp_data = parse_geneExpression(data_folder)
        exp_data.to_csv(os.path.join(output_folder, 'reducedGE.csv'), index = False)
    if not os.path.exists(os.path.join(output_folder, 'reducedCNA.csv')):
        cna_data = parse_CNA(data_folder)
        cna_data.to_csv(os.path.join(output_folder, 'reducedCNA.csv'), index = False)    
        
    clinical_data = pd.read_csv(os.path.join(output_folder, 'clinical_data.csv'))
    exp_data = pd.read_csv(os.path.join(output_folder, 'reducedGE.csv'))
    cna_data = pd.read_csv(os.path.join(output_folder, 'reducedCNA.csv'))
    exp_data = transpose(exp_data)    
    cna_data = transpose(cna_data)
    df1 = generate_Features(exp_data, ['PATIENT_ID'])
    df2 = generate_Features(cna_data, ['PATIENT_ID'])
    #print(exp_data.shape, df1.shape)
    #print(cna_data.shape, df2.shape)
    df3 = pd.merge(clinical_data, df1, on = 'PATIENT_ID')
    df4 = pd.merge(clinical_data, df2, on = 'PATIENT_ID')
    df5 = pd.merge(df3, df2, on = 'PATIENT_ID')
    df6 = remove_missingValues(df3)
    df7 = remove_missingValues(df4)
    df8 = remove_missingValues(df5)
    #print(clinical_data.shape, df1.shape, df3.shape, df6.shape)
    #print(clinical_data.shape, df2.shape, df4.shape, df7.shape)
    #print(df5.shape, df8.shape)
    df6.to_csv(os.path.join(output_folder, 'clinical_GE.csv'), index = False)
    df7.to_csv(os.path.join(output_folder, 'clinical_CNA.csv'), index = False)
    df8.to_csv(os.path.join(output_folder, 'clinical_GE_CNA.csv'), index = False)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
    description = __doc__, formatter_class = argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--Data', type=str, required=True, help='name of data folder')
    parser.add_argument('--Output', type=str, required=True, help='name of output folder')
    args = parser.parse_args()
    main(args)
    
