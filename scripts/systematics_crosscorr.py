import numpy as np
import matplotlib.pyplot as plt
import fitsio
import treecorr
import healpy as hp
import matplotlib.pyplot as plt
import ipdb
import os


def hpRaDecToHEALPixel(ra, dec, nside=4096, nest=False):
    phi = ra * np.pi / 180.0
    theta = (90.0 - dec) * np.pi / 180.0
    hpInd = hp.ang2pix(nside, theta, phi, nest=nest)
    return hpInd

def assign_systematics_weights(catalog, ratag = 'ra', dectag = 'dec', z_tag = 'zredmagic', map_path = './maps',nside=4096, nest=False):
    zbins_hi = np.array([0.35, 0.50, 0.65, 0.80, 0.90])
    zbins_lo = np.array([0.15, 0.35, 0.50, 0.65, 0.80])
    weights = np.zeros_like(catalog[ratag])
    map_filenames = ['w_map_bin0_nside4096_nbins1d_10_2.0sig.fits','w_map_bin1_nside4096_nbins1d_10_2.0sig.fits',
                     'w_map_bin2_nside4096_nbins1d_10_2.0sig.fits','w_map_bin3_nside4096_nbins1d_10_2.0sig.fits',
                     'w_map_bin4_nside4096_nbins1d_10_2.0sig.fits']
    
    map_files = [os.path.join(map_path,ifile) for ifile in map_filenames]
    hpInds = hpRaDecToHEALPixel(catalog[ratag],catalog[dectag],nside=nside,nest=nest)
    for zlo,zhi,ifile in zip(zbins_lo,zbins_hi,map_files):
        these = (catalog[z_tag] >= zlo) & (catalog[z_tag] < zhi)
        weightmap_compressed = fitsio.read(ifile)
        weightmap_full = np.zeros(hp.nside2npix(nside))
        weightmap_full[weightmap_compressed['HPIX']] = weightmap_compressed['VALUE']
        weights[these] = weightmap_full[hpInds[these]]
    return weights


def get_catalogs(zmin=0,zmax=10, weights = True, jacknife = True):
    blg_datafile = "../data/redmagic_lens_v0.5.1_wide_balrog_merged_v1.4_masked.fits"
    blg_data = fitsio.read(blg_datafile)
    blg_data = blg_data[( blg_data['zredmagic'] > zmin ) & (blg_data['zredmagic'] <= zmax)]
    if weights:
        blg_wts = assign_systematics_weights(blg_data,ratag='meas_ra',dectag = 'meas_dec', z_tag = 'zredmagic')
        blg_datacat = treecorr.Catalog(ra = blg_data['meas_ra'], dec = blg_data['meas_dec'], ra_units = 'deg', dec_units = 'deg', w = blg_wts, npatch = 50)
    else:
        blg_datacat = treecorr.Catalog(ra = blg_data['meas_ra'], dec = blg_data['meas_dec'], ra_units = 'deg', dec_units = 'deg', npatch = 50)
    blg_ranfile = "../data/randoms_detection_balrog_merged_v1.4_masked_removedbadtile.fits"
    blg_rancat = treecorr.Catalog(blg_ranfile, ra_col='RA', dec_col= 'DEC',ra_units='deg', dec_units='deg', patch_centers = blg_datacat.patch_centers)
  
        
    rm_randfile = "../data/y3_redmagic_combined_sample_fid_x40_1_randoms.fits"
    rm_rancat = treecorr.Catalog(rm_randfile,ra_col = 'RA', dec_col = 'DEC', ra_units='deg', dec_units='deg', patch_centers = blg_datacat.patch_centers)
    rm_datafile = "../data/y3_gold_2.2.1_wide_sofcol_run_redmapper_v0.5.1_combined_hd3_hl2_sample_weighted2.0sig.fits"
    rm_data = fitsio.read(rm_datafile)
    rm_data = rm_data[( rm_data['ZREDMAGIC'] > zmin) & (rm_data['ZREDMAGIC'] <= zmax)]
    if weights:
        rm_datacat = treecorr.Catalog(ra = rm_data['RA'], dec = rm_data['DEC'], ra_units = 'deg', dec_units = 'deg', w_col = rm_data['weight'], patch_centers = blg_datacat.patch_centers)
        #rm_datacat = treecorr.Catalog(rm_datafile,ra_col = 'RA', dec_col = 'DEC', ra_units='deg', dec_units='deg',w_col = 'weight',)
    else:
        rm_datacat = treecorr.Catalog(ra = rm_data['RA'], dec = rm_data['DEC'], ra_units = 'deg', dec_units = 'deg', patch_centers = blg_datacat.patch_centers)
        #rm_datacat = treecorr.Catalog(rm_datafile,ra_col = 'RA', dec_col = 'DEC', ra_units='deg', dec_units='deg')
    return blg_datacat, rm_datacat, blg_rancat, rm_rancat

def cross_correlate(cat1,cat2,ran1,ran2):
    D1_D2 = treecorr.NNCorrelation(min_sep = 1, max_sep = 200, nbins = 10,sep_units = 'arcmin',var_method = 'jackknife')
    D1_D2.process(cat1, cat2)  
    D1_R2 = treecorr.NNCorrelation(min_sep = 1, max_sep = 200, nbins = 10,sep_units = 'arcmin',var_method = 'jackknife')
    D1_R2.process(cat1, ran2)
    R1_D2 = treecorr.NNCorrelation(min_sep = 1, max_sep = 200, nbins = 10,sep_units = 'arcmin',var_method = 'jackknife')
    R1_D2.process(ran1, cat2)
    R1_R2 = treecorr.NNCorrelation(min_sep = 1, max_sep = 200, nbins = 10,sep_units = 'arcmin',var_method = 'jackknife')
    R1_R2.process(ran1, ran2)
    xi_d, var_d = D1_D2.calculateXi(R1_R2,D1_R2, R1_D2)
    cov_jk = D1_D2.estimate_cov(method='jackknife')
    logr = D1_D2.meanlogr
    theta = np.exp(logr)
    return theta, xi_d, np.sqrt(np.diag(cov_jk))




def main():
    zbins_hi = np.array([0.35, 0.50, 0.65, 0.80, 0.90])
    zbins_lo = np.array([0.15, 0.35, 0.50, 0.65, 0.80])

    for zlo,zhi in zip(zbins_lo,zbins_hi):
        
        blg, rm, blg_ran, rm_ran = get_catalogs(weights = True, zmin=zlo, zmax = zhi)
        blg_nw, rm_nw, blg_ran_nw, rm_ran_nw = get_catalogs(weights = False, zmin=zlo, zmax = zhi)

        theta, xi, sigma = cross_correlate(blg, rm, blg_ran, rm_ran)
        theta_nw, xi_nw, sigma_nw = cross_correlate(blg_nw, rm_nw, blg_ran_nw, rm_ran_nw)

        # So try plotting the cross-correlation:
    
        plt.errorbar(theta, theta*xi,theta*sigma,label='weights')
        plt.errorbar(theta_nw, theta_nw*xi_nw, theta_nw*sigma_nw,label='no weights')
        plt.legend()
        plt.xscale('log')
        plt.axhline(0,color='grey',linestyle='--',alpha=0.5)
        plt.xlabel('angle (arcmin)')
        plt.ylabel('theta * BxRM')
        plt.tight_layout()
        plt.savefig(f'./plots/balrog_redmagic_crosscorr-{zlo:.03}_{zhi:03}.png')
        plt.clf()
    ipdb.set_trace()

if __name__ == '__main__':
    main()
