import mne
from osl_ephys.source_recon import rhino, beamforming, parcellation
import pickle
import argparse
import os

def process_parcellated_raw(
    filename,           # Raw FIF file
    subjects_dir,       # FreeSurfer subjects directory
    subject,            # Subject identifier
    parcellation_fname, # Parcellation NIfTI file
    labels_fname,       # Pickle file with labels
    plot=True           # Whether to plot the final parc_raw
):
    """
    Load raw data, apply LCMV beamforming, parcellate source-space timeseries,
    convert to MNE Raw object, and optionally plot.
    
    Returns:
        parc_raw : mne.io.Raw
            Parcellated raw data with magnetometer channels and restored annotations.
    """
    
    # -----------------------
    # Load raw data
    # -----------------------
    clean = mne.io.read_raw_fif(filename, preload=True)

    # -----------------------
    # Compute data rank
    # -----------------------
    rank = mne.compute_rank(clean)
    rank['mag'] = rank['mag'] - 2
    print(rank)

    # -----------------------
    # Make LCMV beamformer filters
    # -----------------------
    filters = beamforming.make_lcmv(
        subjects_dir,
        subject,
        clean,
        {'mag'},
        pick_ori="max-power-pre-weight-norm",
        reduce_rank=True,
        rank=rank,
    )
    print("Applying beamformer spatial filters")

    # -----------------------
    # Apply beamformer
    # -----------------------
    stc = beamforming.apply_lcmv(clean, filters, reject_by_annotation=None)

    # -----------------------
    # Transform timeseries to MNI space
    # -----------------------
    recon_timeseries_mni, _, recon_coords_mni, _ = beamforming.transform_recon_timeseries(
        subjects_dir,
        subject,
        recon_timeseries=stc.data,
        reference_brain="mni"
    )
    print("Dimensions of reconstructed timeseries in MNI space (dipoles x tpts):", recon_timeseries_mni.shape)

    # -----------------------
    # Load parcellation and labels
    # -----------------------
    if plot:
        parcellation.plot_parcellation(parcellation_fname)
    
    with open(labels_fname, 'rb') as f:
        labels = pickle.load(f)

    # -----------------------
    # Parcellate timeseries
    # -----------------------
    parcel_ts, _, _ = parcellation.parcellate_timeseries(
        parcellation_fname,
        recon_timeseries_mni,
        recon_coords_mni,
        "spatial_basis",
        None
    )

    # -----------------------
    # Handle annotations
    # -----------------------
    annotations_copy = clean.annotations.copy()
    clean.set_annotations(None)

    # -----------------------
    # Convert parcellated timeseries to MNE Raw
    # -----------------------
    parc_raw = parcellation.convert2mne_raw(parcel_ts, clean, labels)

    # Restore annotations safely
    try:
        parc_raw.set_annotations(annotations_copy)
    except ValueError:
        print("Warning: could not restore annotations (time mismatch)")

    # Set all channels to magnetometers
    parc_raw.set_channel_types({ch: 'mag' for ch in parc_raw.info['ch_names']})

    # -----------------------
    # Optional plotting
    # -----------------------
    if plot:
        parc_raw.plot(scalings='auto')

    return parc_raw


# ========================
# COMMAND-LINE INTERFACE
# ========================
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process MEG data, beamform, parcellate, and convert to MNE Raw.")
    parser.add_argument("filename", help="Raw FIF file")
    parser.add_argument("subjects_dir", help="FreeSurfer subjects directory")
    parser.add_argument("subject", help="Subject identifier")
    parser.add_argument("parcellation_fname", help="Parcellation NIfTI file")
    parser.add_argument("labels_fname", help="Pickle file containing labels")
    parser.add_argument("--plot", action="store_true", help="Plot the final parcellated Raw object")
    
    args = parser.parse_args()
    
    parc_raw = process_parcellated_raw(
        filename=args.filename,
        subjects_dir=args.subjects_dir,
        subject=args.subject,
        parcellation_fname=args.parcellation_fname,
        labels_fname=args.labels_fname,
        plot=args.plot
    )
