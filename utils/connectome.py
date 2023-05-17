# Functions to build and process connectomes

import mne
import numpy as np
import scipy
from sklearn import preprocessing
import gdist               # for geodesic computations
from .mathUtil import fwhm2sigma, max_smoothing_distance, diagonal_stack_sparse_matrices
from .readers import read_source_space, read_tractography



# Functions to handle source space
# --------------------------------

def build_source_space(bids_dir, subject):
    """
    Uses the setup_source_space function from MNE to generate a source space from the 
    freesurfer folder.

    Parameters:
        bids_dir: Path
            path to VEPCON data set folder
        
        subject: str
            name of subject

    Returns:
        src: mne.source_space.SourceSpaces
        source space
    """

    subjects_dir = bids_dir / 'derivatives' / 'freesurfer-7.1.1'

    src = mne.setup_source_space(subject, subjects_dir=subjects_dir)

    return src


# Functions for geodesic surface smoothing
# -------------------------------


def local_geodesic_distances(vertices, faces, max_distance):
    """ 
    Generate a local geodesic distance symmetric matrix.
    The geodesic distance between two vertices is the shortest distance between the two
    vertices on a specific surface.

    Parameters:
        vertices: ndarray (n,3)
            (x,y,z) coordinates of the mesh's vertices

        faces: ndarray (m,3)
            faces of the mesh as index triplets into vertices
            
        max_distance: float
            maximum distance for local measurements

    Returns:
        distances : scipy.sparse.csr (n,n)
            geodesic distance matrix
    """

    # Make sure the inputs have correct type
    vertices     = vertices.astype(np.float64)
    faces        = faces.astype(np.int32)

    # compute geodesic distances
    distances = gdist.local_gdist_matrix(vertices, faces, max_distance)

    # make matrix symmetric
    # gdist.local_gdist_matrix does not compute the distance similarly when going
    # i->j or j->i. Therefore we make it symmetric by choosing the smallest distance
    # between the two computed.
    distances = distances.minimum(distances.T)       # Element-wise minimum between this and another matrix.

    distances.eliminate_zeros()                     # removes explicit zeros if any 
    distances.tocsr()                               # convert to scipy.sparse compressed sparse row matrix (instead of column)

    return distances


def local_distances_to_smoothing_coefficients(local_distance, sigma):
    """
    Generate an sparse asymmetric coefficient matrix

    Takes a sparse local distance symmetric matrix (CSR) as input,
    Generates an assymetric coefficient sparse matrix where each
    row i, has the coefficient for smoothing a signal from node i,
    therefore, each row sum is unit (1). sigma comes from the smoothing
    variance.

    Parameters:
        local_distance: scipy.sparse.csr (n,n)
            matrix showing local geodesic distances between vertices
    
        sigma: float
            sigma parameter for gaussian distribution

    Return:

    """

    # apply gaussian transform on geodesic matrix
    output = -(local_distance.power(2) / (2 * (sigma **2)))
    np.exp(output.data, out=output.data)

    # add ones to the diagonal 
    output += scipy.sparse.eye(output.shape[0], dtype=output.dtype).tocsr()

    # normalize rows of matrix
    output = preprocessing.normalize(output, norm='l1')

    return output


def get_streamline_incidence(startDists, startIndices, endDists, endIndices, nodeCount, threshold=2):
    """
    Delete streamlines with end points further than 'threshold' from closest source
    point (vertex).
    Return one incidence matrix for starting points and one incidence matrix for
    ending points

    Parameters:
        startDists: ndarray
            distances between starting point of streamlines and their closest source point
        
        startIndices: ndarray
            indices for source point closest to starting point of streamlines

        endDists: ndarray
            distances between ending point of streamlines and their closest source point

        endIndices: ndarray
            indices for source point closest to ending point of streamlines
        
        nodeCount: int
            number of source points (nodes)

        threshold: float
            maximum accepted distance (in mm) to closest source point.   

    Returns:
        startIncidence: scipy.sparse.csr
            incidence matrix for starting points of streamlines

        endIncidence: scipy.sparse.csr
            incidence matrix for ending points of streamlines
    """

    # mask points that are further than 'threshold' from closest source point
    mask = ~ ((startDists > threshold) | (endDists > threshold))

    # filter streamline points
    startFilt = startIndices[mask]
    endFilt   = endIndices[mask]

    # Save links in sparse incidence matrix
    startIncidence = scipy.sparse.dok_matrix((nodeCount, mask.sum()))
    endIncidence   = scipy.sparse.dok_matrix((nodeCount, mask.sum()))

    for i in range(len(startFilt)):
        startIncidence[(startFilt[i], i)] = 1
        endIncidence[(endFilt[i], i)]     = 1

    return (startIncidence.tocsr(), endIncidence.tocsr())




def get_smoothed_adjacency_from_unsmoothed_incidence(startIncidence, endIncidence, localSmoothingCoefficients):
    """
    Return a smoothed sparse adjacency matrix from the two halves of the incidence matrix.

    The smoothing is done at the network level, that is, the incidence matrices are 
    smoothed before creation of the adjacency.

    Parameters:
        startIncidence: scipy.sparse.csr
            incidence matrix for starting points of streamlines
        
        endIncidence: scipy.sparse.csr
            incidence matrix for ending points of streamlines

        localSmoothingCoefficients: scipy.sparse.csr
            smoothing coefficients for local connectivity

    Return:
        adjacency: scipy.sparse.csr
            adjacency matrix for neural connectivity (connectome)
    """

    startIncidenceSmoothed = startIncidence.T.dot(localSmoothingCoefficients).T
    endIncidenceSmoothed   = endIncidence.T.dot(localSmoothingCoefficients).T

    adjacency = startIncidenceSmoothed.dot(endIncidenceSmoothed.T)

    return adjacency + adjacency.T      # make sure adjacency matrix is symmetric


# function implementing connectome building
# -----------------------------------------

def build_connectome(source, tractography, cras, affMat):
    """
    Builds a connectome from a source space and a tractography

    Parameters:
        source: mne.sourceSpace
            source space

        tractography: nibabel.streamlines
            tractography

        cras: ndarray
            Talairach transformation

        affMat:
            affine transformation

    Returns:
        connectome: ndarray
            connectome (neural connectivity adjacency matrix)

        ids: ndarray
            ????

    """

    # Preprocessing - source space
    # ----------------------------

    sources = {}
    for index, hemisphere in enumerate(['left', 'right']):

        # extract source
        src = source[index]['rr']

        # project from meter into millimeter space
        src *= 1000

        # project into Talairach space
        src += cras 

        # project surface to 'diffusion space' (apply ANTs affine transformation)
        src = np.concatenate((src, np.zeros((len(src), 1))), 1)
        src = np.linalg.inv(affMat) @ src.T
        src = src[:3].T

        # only select 'used' sources 
        src = src[source[index]['inuse'] == 1]

        sources[hemisphere] = src 

    
    # Process - aggregate tractography points into source points
    # ----------------------------------------------------------

    # get source positions
    surfaceXYZ = np.concatenate((sources['left'], sources['right']), 0)

    # get tractography end-to-end points
    starts = np.array([stream[0] for stream in tractography.streamlines]) / tractography.header["voxel_sizes"][np.newaxis, :]
    ends   = np.array([stream[-1] for stream in tractography.streamlines]) / tractography.header["voxel_sizes"][np.newaxis, :]

    # build nearest neighbor classifier
    kd_tree = scipy.spatial.cKDTree(surfaceXYZ)

    # classify tractography points to nearest source points
    startDists, startIndices = kd_tree.query(starts)
    endDists, endIndices     = kd_tree.query(ends)

    # compute incidence matrices for starting and ending streamline points
    incidences = get_streamline_incidence(startDists, startIndices, endDists, endIndices, len(surfaceXYZ))


    # Process- smoothen surface (= add local geodesic connectivity)
    # -------------------------------------------------------

    # * compute local geodesic distances *

    # parameters for computing local geodesic distances
    sigma        = fwhm2sigma(12)
    epsilon      = 1e-07
    max_distance = max_smoothing_distance(sigma, epsilon, 2)

    # compute geodesic distances
    localDistances = {} 
    for index, hemisphere in enumerate(['left', 'right']):

        # extract mesh vertices
        vertices = sources[hemisphere]

        # find vertex number and associated array index 
        vertexNumber = source[index]['vertno']               # 'vertno' stands for vertex number
        indices      = np.arange(len(vertexNumber))          # index of each vertex

        # map each vertex to its array index
        mapping = {vertexNumber[i]:i for i in indices}

        # extract mesh faces and map to the vertex indices
        faces = source[index]['use_tris']
        faces = np.array([[mapping[v] for v in face] for face in faces])

        # compute local geodesic distances from vertices and faces
        localDistances[hemisphere] = local_geodesic_distances(vertices, faces, max_distance)

    # diagonaly stack geodesic connectivity matrices of both hemisphers
    geoAdj = diagonal_stack_sparse_matrices(localDistances['left'], localDistances['right'])

    # * Apply smoothing *

    # compute smoothing coefficients
    smoothingCoefs = local_distances_to_smoothing_coefficients(geoAdj, sigma)

    # apply smoothing
    adjacency = get_smoothed_adjacency_from_unsmoothed_incidence(*incidences, smoothingCoefs)
    ids       = adjacency.toarray().sum(0) == 0

    
    # Process - build connectome from previous adjacency matrix
    # ---------------------------------------------------------

    connectome = adjacency.toarray()[ids == 0][:, ids==0]
    connectome = connectome * (1 - np.eye(len(connectome)))

    return connectome, ids


def build_subject_connectome(bids_dir, subject, src_dir=None):
    """
    Build a connectome for specified subject

    Parameters:
        bids_dir: Path
            path to VEPCON data set folder
        
        subject: str
            name of subject

        src_dir: Path
            path to dataset containing saved source spaces

    Returns:
        source: mne.SourceSpace
            source space

        connectome: ndarray
            connectome (neural connectivity adjacency matrix)

        ids: ndarray
            ???
    """

    # Read source space if src_dir not None else build it
    if src_dir is None:
        source = build_source_space(bids_dir, subject)
    else:
        source = read_source_space(src_dir, subject)

    # Read tractography
    tractography = read_tractography(bids_dir, subject)

    # Load Talairach tansformation
    talTrsf = bids_dir / 'derivatives' / 'freesurfer-7.1.1' / subject / 'mri' / 'transforms' / 'talairach.lta'
    cras    = np.asarray(list(map(float, np.loadtxt(talTrsf, str, skiprows=20, max_rows=1)[2:5])))

    # Load affine transformation
    affTrsf = bids_dir / 'derivatives' / 'cmp-v3.0.3' / subject / 'xfm' / 'final0GenericAffine.mat'
    affMat  = scipy.io.loadmat(affTrsf)['AffineTransform_double_3_3'].reshape(3, 4, order='F')
    affMat  = np.concatenate((affMat, np.array([[0, 0, 0, 1]])))

    # * build connectome *
    connectome, ids = build_connectome(source, tractography, cras, affMat)

    return source, connectome, ids