from io import StringIO

import pandas as pd
import requests


def get_ppi_encoder(chosen_genes: list[str], score: str = 'score', threshold: float = 0.0):
    """Get PPI for chosen genes

    Args:
        chosen_genes (list[str]): list of chosen genes

    Returns:
        ppi (pd.DataFrame): PPI network
    """
    gene_encoder = dict(zip(chosen_genes, range(len(chosen_genes))))

    res_text = _get_ppi_network_from_string(chosen_genes)

    ppi = pd.read_csv(StringIO(res_text), sep='\t')
    ppi = ppi[['preferredName_A', 'preferredName_B', 'score']]
    ppi.drop_duplicates(inplace=True)
    ppi = ppi[ppi[score] >= threshold]
    #ppi[['src', 'dst']] = ppi[['preferredName_A', 'preferredName_B']].map(lambda x: gene_encoder[x])
    ppi[['src', 'dst']] = ppi[['preferredName_A', 'preferredName_B']].applymap(lambda x: gene_encoder[x]) #different versions of pandas
    return ppi


def _get_ppi_network_from_string(gene_list: list[str]):
    """Get PPI network from STRING API.

    Args:
        gene_list (list[str]): list of genes

    Returns:
        res_text (str): response from STRING API in text format
    """
    string_api_url = 'https://string-db.org/api'
    request_url = '/'.join([string_api_url, 'tsv', 'network'])

    # Speed up the process by getting the identifier first.
    if len(gene_list) > 100:
        gene_list = _get_identifier(gene_list, string_api_url).values()

    params = {
        'identifiers': '%0d'.join(gene_list),           # Genes
        'species': 9606,                                # Species NCBI identifier for homo sapiens
        'caller_identity': 'www.idssp.ee.ntu.edu.tw'    # App name
    }

    # Link to the graph.
    # request_url = '/'.join([string_api_url, 'tsv-no-header', 'get_link'])
    # response = requests.post(request_url, data=params)

    response = requests.post(request_url, data=params)
    if response.status_code != 200:
        raise ConnectionError(f'Responce from STRING: {response.status_code}')
    return response.text


def _get_identifier(gene_list: list[str], string_api_url: str):
    """Each gene has corresponding identifiers in STRING. Get the best one out of them.

    Args:
        gene_list (list[str]): list of genes
        string_api_url (str): String API

    Returns:
        identifiers (dict[str, str]): corresponding identifiers
    """
    params = {
        'identifiers': '\r'.join(gene_list),            # Genes
        'species': 9606,                                # Species NCBI identifier Homo Sapiens
        'limit': 1,                                     # Return only one (best) identifier per protein
        'caller_identity': 'www.idssp.ee.ntu.edu.tw'    # App name
    }
    request_url = '/'.join([string_api_url, 'tsv-no-header', 'get_string_ids'])
    response = requests.post(request_url, data=params)

    if response.status_code != 200:
        raise ConnectionError(f'Responce from STRING: {response.status_code}')

    identifiers: dict[str, str] = {}
    for line in response.text.strip().split('\n'):
        counter, identifier, species_id, species, gene, meaning = line.split('\t')
        identifiers[gene] = identifier
    if set(identifiers.keys()) != set(gene_list):
        raise ValueError('Cannot get correct identifiers.')
    return identifiers
