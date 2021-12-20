#Code Copied From https://github.com/pkomiske/EnergyFlow/blob/master/energyflow
from emd_utils import p4s_from_ptyphims
import ot
from ot.lp import emd_c, check_result
from scipy.spatial.distance import _distance_wrap # ot imports scipy anyway
import numpy as np

def _check_params(norm, gdim, phi_col, measure, coords, empty_policy):

    # check norm
    if norm is None:
        raise ValueError("'norm' cannot be None")

    # check phi_col
    if phi_col < 1:
        raise ValueError("'phi_col' cannot be smaller than 1")

    # check gdim
    if gdim is not None:
        if gdim < 1:
            raise ValueError("'gdim' must be greater than or equal to 1")
        if phi_col > gdim + 1:
            raise ValueError("'phi_col' must be less than or equal to 'gdim'")

    # check measure
    if measure not in {'euclidean', 'spherical'}:
        raise ValueError("'measure' must be one of 'euclidean', 'spherical'")

    # check coords
    if coords not in {'hadronic', 'cartesian'}:
        raise ValueError("'coords' must be one of 'hadronic', 'cartesian'")

    # check empty_policy
    if not (isinstance(empty_policy, (int, float)) or empty_policy == 'error'):
        raise ValueError("'empty_policy' must be a number or 'error'")

# process events for EMD calculation
two_pi = 2*np.pi
def _process_for_emd(event, norm, gdim, periodic_phi, phi_col, 
                     mask, R, hadr2cart, euclidean, error_on_empty):
    
    # ensure event is at least a 2d numpy array
    event = np.atleast_2d(event) if gdim is None else np.atleast_2d(event)[:,:(gdim+1)]

    # if we need to map hadronic coordinates to cartesian ones
    if hadr2cart:
        event = p4s_from_ptyphims(event)

    # select the pts and coords
    pts, coords = event[:,0], event[:,1:]

    # norm vectors if spherical
    if not euclidean:

        # special case for three dimensions (most common), twice as fast
        if coords.shape[1] == 3:
            coords /= np.sqrt(coords[:,0]**2 + coords[:,1]**2 + coords[:,2]**2)[:,None]
        else:
            coords /= np.sqrt(np.sum(coords**2, axis=1))[:,None]

    # handle periodic phi (only applicable if using euclidean)
    elif periodic_phi:
        if phi_col >= event.shape[1] - 1:
            evgdim = str(event.shape[1] - 1)
            raise ValueError("'phi_col' cannot be larger than the ground space "
                             'dimension, which is ' + evgdim + ' for one of the events')
        coords[:,phi_col] %= two_pi

    # handle masking out particles farther than R away from origin
    if mask:

        # ensure contiguous coords for scipy distance function
        coords = np.ascontiguousarray(coords, dtype=np.double)
        origin = np.zeros((1, coords.shape[1]))

        # calculate distance from origin
        rs = _cdist(origin, coords, euclidean, periodic_phi, phi_col)[0]
        rmask = (rs <= R)

        # detect when masking actually needs to occur
        if not np.all(rmask):
            pts, coords = pts[rmask], coords[rmask]

    # check that we have at least one particle
    if pts.size == 0:
        if error_on_empty:
            raise ValueError('empty event encountered, must have at least one particle')
        else:
            return (None, None)

    # handle norming pts or adding extra zeros to event
    if norm:
        pts = pts/pts.sum()
    elif norm is None:
        pass
    else:
        coords = np.vstack((coords, np.zeros(coords.shape[1])))
        pts = np.concatenate((pts, np.zeros(1)))

    return (np.ascontiguousarray(pts, dtype=np.double), 
            np.ascontiguousarray(coords, dtype=np.double))

# faster than scipy's cdist function because we can avoid their checks
def _cdist(X, Y, euclidean, periodic_phi, phi_col):
    if euclidean:
        if periodic_phi:

            # delta phis (taking into account periodicity)
            # aleady guaranteed for values to be in [0, 2pi]
            d_phis = np.pi - np.abs(np.pi - np.abs(X[:,phi_col,None] - Y[:,phi_col]))

            # split out common case of having only one other dimension
            if X.shape[1] == 2:
                non_phi_col = 1 - phi_col
                d_ys = X[:,non_phi_col,None] - Y[:,non_phi_col]
                out = np.sqrt(d_ys**2 + d_phis**2)

            # general case
            else:
                non_phi_cols = [i for i in range(X.shape[1]) if i != phi_col]
                d_others2 = (X[:,None,non_phi_cols] - Y[:,non_phi_cols])**2
                out = np.sqrt(d_others2.sum(axis=-1) + d_phis**2)

        else:
            out = np.empty((len(X), len(Y)), dtype=np.double)
            _distance_wrap.cdist_euclidean_double_wrap(X, Y, out)

    # spherical measure
    else:

        # add min/max conditions to ensure valid input
        out = np.arccos(np.fmax(np.fmin(np.tensordot(X, Y, axes=(1, 1)), 1.0), -1.0))

    return out

# helper function for pool imap
def _emd4map(x):
    (i, j), params = x
    return _emd(_X0[i], _X1[j], *params)

# internal use only by emds, makes assumptions about input format
def _emd(ev0, ev1, R, no_norm, beta, euclidean, n_iter_max,
         periodic_phi, phi_col, empty_policy):

    pTs0, coords0 = ev0
    pTs1, coords1 = ev1

    if pTs0 is None or pTs1 is None:
        return empty_policy

    thetas = _cdist(coords0, coords1, euclidean, periodic_phi, phi_col)/R

    # implement angular exponent
    if beta != 1:
        thetas **= beta

    # extra particles (with zero pt) already added if going in here
    rescale = 1.0
    if no_norm:
        pT0, pT1 = pTs0.sum(), pTs1.sum()
        pTdiff = pT1 - pT0
        if pTdiff > 0:
            pTs0[-1] = pTdiff
        elif pTdiff < 0:
            pTs1[-1] = -pTdiff
        thetas[:,-1] = 1.0
        thetas[-1,:] = 1.0

        # change units for numerical stability
        rescale = max(pT0, pT1)

    # compute the emd with POT
    _, cost, _, _, result_code = emd_c(pTs0/rescale, pTs1/rescale, thetas, n_iter_max)
    check_result(result_code)

    # important! must reset extra particles to have pt zero
    if no_norm:
        pTs0[-1] = pTs1[-1] = 0

    return cost * rescale


def emd_pot(ev0, ev1, R=1.0, norm=False, beta=1.0, measure='euclidean', coords='hadronic',
                     return_flow=False, gdim=None, mask=False, n_iter_max=100000,
                     periodic_phi=False, phi_col=2, empty_policy='error'):
    
    r"""Compute the EMD between two events using the Python Optimal
    Transport library.
    **Arguments**
    - **ev0** : _numpy.ndarray_
        - The first event, given as a two-dimensional array. The event is 
        assumed to be an `(M,1+gdim)` array of particles, where `M` is the 
        multiplicity and `gdim` is the dimension of the ground space in 
        which to compute euclidean distances between particles (as specified
        by the `gdim` keyword argument. The zeroth column is assumed to be
        the energies (or equivalently, the transverse momenta) of the 
        particles. For typical hadron collider jet applications, each 
        particle will be of the form `(pT,y,phi)` where  `y` is the rapidity
        and `phi` is the azimuthal angle.
    - **ev1** : _numpy.ndarray_
        - The other event, same format as `ev0`.
    - **R** : _float_
        - The R parameter in the EMD definition that controls the relative 
        importance of the two terms. Must be greater than or equal to half 
        of the maximum ground distance in the space in order for the EMD 
        to be a valid metric satisfying the triangle inequality.
    - **beta** : _float_
        - The angular weighting exponent. The internal pairwsie distance
        matrix is raised to this power prior to solving the optimal
        transport problem.
    - **norm** : _bool_
        - Whether or not to normalize the pT values of the events prior to 
        computing the EMD.
    - **measure** : _str_
        - Controls which metric is used to calculate the ground distances
        between particles. `'euclidean'` uses the euclidean metric in
        however many dimensions are provided and specified by `gdim`.
        `'spherical'` uses the opening angle between particles on the
        sphere (note that this is not fully tested and should be used
        cautiously).
    - **coords** : _str_
        - Only has an effect if `measure='spherical'`, in which case it
        controls if `'hadronic'` coordinates `(pT,y,phi,[m])` are expected
        versus `'cartesian'` coordinates `(E,px,py,pz)`.
    - **return_flow** : _bool_
        - Whether or not to return the flow matrix describing the optimal 
        transport found during the computation of the EMD. Note that since
        the second term in Eq. 1 is implemented by including an additional 
        particle in the event with lesser total pT, this will be reflected 
        in the flow matrix.
    - **gdim** : _int_
        - The dimension of the ground metric space. Useful for restricting
        which dimensions are considered part of the ground space. Can be
        larger than the number of dimensions present in the events (in
        which case all dimensions will be included). If `None`, has no
        effect.
    - **mask** : _bool_
        - If `True`, ignores particles farther than `R` away from the
        origin.
    - **n_iter_max** : _int_
        - Maximum number of iterations for solving the optimal transport 
        problem.
    - **periodic_phi** : _bool_
        - Whether to expect (and therefore properly handle) periodicity
        in the coordinate corresponding to the azimuthal angle $\phi$.
        Should typically be `True` for event-level applications but can
        be set to `False` (which is slightly faster) for jet applications
        where all $\phi$ differences are less than or equal to $\pi$.
    - **phi_col** : _int_
        - The index of the column of $\phi$ values in the event array.
    - **empty_policy** : _float_ or `'error'`
        - Controls behavior if an empty event is passed in. When set to
        `'error'`, a `ValueError` is raised if an empty event is
        encountered. If set to a float, that value is returned is returned
        instead on an empty event.
    **Returns**
    - _float_
        - The EMD value.
    - [_numpy.ndarray_], optional
        - The flow matrix found while solving for the EMD. The `(i,j)`th 
        entry is the amount of `pT` that flows between particle i in `ev0`
        and particle j in `ev1`.
    """

    # parameter checks
    _check_params(norm, gdim, phi_col, measure, coords, empty_policy)
    euclidean = (measure == 'euclidean')
    hadr2cart = (not euclidean) and (coords == 'hadronic')
    error_on_empty = (empty_policy == 'error')

    # handle periodicity
    phi_col_m1 = phi_col - 1

    # process events
    args = (None, gdim, periodic_phi, phi_col_m1, 
            mask, R, hadr2cart, euclidean, error_on_empty)
    pTs0, coords0 = _process_for_emd(ev0, *args)
    pTs1, coords1 = _process_for_emd(ev1, *args)

    if pTs0 is None or pTs1 is None:
        if return_flow:
            return empty_policy, np.zeros((0,0))
        else:
            return empty_policy

    pT0, pT1 = pTs0.sum(), pTs1.sum()

    # if norm, then we normalize the pts to 1
    if norm:
        pTs0 /= pT0
        pTs1 /= pT1
        thetas = _cdist(coords0, coords1, euclidean, periodic_phi, phi_col_m1)/R
        rescale = 1.0

    # implement the EMD in Eq. 1 of the paper by adding an appropriate extra particle
    else:
        pTdiff = pT1 - pT0
        if pTdiff > 0:
            pTs0 = np.hstack((pTs0, pTdiff))
            coords0_extra = np.vstack((coords0, np.zeros(coords0.shape[1], dtype=np.float64)))
            thetas = _cdist(coords0_extra, coords1, euclidean, periodic_phi, phi_col_m1)/R
            thetas[-1,:] = 1.0

        elif pTdiff < 0:
            pTs1 = np.hstack((pTs1, -pTdiff))
            coords1_extra = np.vstack((coords1, np.zeros(coords1.shape[1], dtype=np.float64)))
            thetas = _cdist(coords0, coords1_extra, euclidean, periodic_phi, phi_col_m1)/R
            thetas[:,-1] = 1.0

        # in this case, the pts were exactly equal already so no need to add a particle
        else:
            thetas = _cdist(coords0, coords1, euclidean, periodic_phi, phi_col_m1)/R

        # change units for numerical stability
        rescale = max(pT0, pT1)

    # implement angular exponent
    if beta != 1:
        thetas **= beta

    G, cost, _, _, result_code = emd_c(pTs0/rescale, pTs1/rescale, thetas, n_iter_max, 1)
    check_result(result_code)

    # need to change units back
    if return_flow:
        G *= rescale
        return cost * rescale, G
    else:
        return cost * rescale