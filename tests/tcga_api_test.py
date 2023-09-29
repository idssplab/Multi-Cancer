from utils import set_random_seed
from utils.api.tcga_api import (download_file, download_files,
                                get_filters_result_from_case,
                                get_filters_result_from_file,
                                get_filters_result_from_project,
                                get_metadata_from_case, get_metadata_from_file,
                                get_metadata_from_project)

SEED = 1126
set_random_seed(SEED)

if __name__ == '__main__':
    simple_test_filters = {
        '=': {'cases.demographic.gender': ['male']}
    }

    complex_test_filters = {
        'and': [
            {'in': {'cases.submitter_id': ['TCGA-CK-4948', 'TCGA-D1-A17N', 'TCGA-4V-A9QX', 'TCGA-4V-A9QM']}},
            {'=': {'files.data_type': 'Gene Expression Quantification'}}
        ]
    }

    project_test_filters = {
        '=': {'program.name': 'TCGA'}
    }

    project_case_test_filters = {
        'and': [
            {'=': {'project.project_id': 'TCGA-BRCA'}},
            {'=': {'files.access': 'open'}},
            {'=': {'files.data_type': 'Gene Expression Quantification'}},
            {'=': {'files.experimental_strategy': 'RNA-Seq'}},
            {'=': {'files.data_format': 'TSV'}},
            {'or': [
                {'=': {'demographic.vital_status': 'Alive'}},
                {'and': [
                    {'=': {'demographic.vital_status': 'Dead'}},
                    {'not': {'diagnoses.days_to_diagnosis': 'missing'}},
                    {'not': {'demographic.days_to_death': 'missing'}}
                ]}
            ]}
        ]
    }

    file_test_filters = {
        'and': [
            {'=': {'cases.project.project_id': 'TCGA-BRCA'}},
            {'=': {'access': 'open'}},
            {'or': [
                {'and': [
                    {'=': {'data_type': 'Clinical Supplement'}},
                    {'=': {'data_format': 'BCR XML'}}
                ]},
                {'and': [
                    {'=': {'data_type': 'Gene Expression Quantification'}},
                    {'=': {'experimental_strategy': 'RNA-Seq'}},
                    {'=': {'data_format': 'TSV'}}
                ]}
            ]}
        ]
    }

    cases_test_filters = {'and': [{'in': {'submitter_id': ['TCGA-BH-A0EA']}}]}

    print(get_metadata_from_project(
        project_id='TARGET-NBL',
        expand=['summary', 'summary.experimental_strategies', 'summary.data_categories']
    ))
    print(get_metadata_from_case(case_id='1f601832-eee3-48fb-acf5-80c4a454f26e', expand='diagnoses'))
    print(get_metadata_from_file(file_id='874e71e0-83dd-4d3e-8014-10141b49f12c'))

    print(get_filters_result_from_project(filters=project_test_filters, sort='summary.case_count:asc'))
    print(get_filters_result_from_case(filters=simple_test_filters, sort='demographic.gender:asc'))
    print(get_filters_result_from_case(filters=complex_test_filters, fields=['case_id']))
    print(get_filters_result_from_case(filters=project_case_test_filters))
    print(get_filters_result_from_case(filters=cases_test_filters))
    print(get_filters_result_from_file(filters=file_test_filters, fields=['cases.case_id', 'file_name', 'file_id']))

    # download_file(file_id='5b2974ad-f932-499b-90a3-93577a9f0573', extract_directory='Data')
    # download_file(file_id='5b2974ad-f932-499b-90a3-93577a9f0573', extract_directory='Data', method='POST')
    # download_file(file_id='7efc039a-fde3-4bc1-9433-2fc6b5e3ffa5', extract_directory='Data', related_files=True)
    # download_file(file_id='7efc039a-fde3-4bc1-9433-2fc6b5e3ffa5', extract_directory='Data', method='POST',
    #               related_files=True)
    # download_files(
    #     file_ids=['e3228020-1c54-4521-9182-1ea14c5dc0f7', '18e1e38e-0f0a-4a0e-918f-08e6201ea140'],
    #     extract_directory='Data'
    # )
