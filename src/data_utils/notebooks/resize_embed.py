import pandas as pd
from pathlib import Path
from pydicom import dcmread
from skimage.io import imsave
from skimage.transform import resize
from tqdm import tqdm
import numpy as np
import multiprocessing as mp
import ast


IMG_SIZE = 512
out_dir = Path(f"/home/moritz/data_huge/embed_small/processed_{IMG_SIZE}x{IMG_SIZE}")
out_dir.mkdir(exist_ok=True)
base_dir = Path("/home/moritz/data_huge/embed_small/tables")
data_dir = Path("/home/moritz/data_huge/embed_small/images")
data_df = pd.read_csv("/home/moritz/repositories/fair_leak/data/csv/embed_all.csv")

# adapted from https://github.com/Emory-HITI/EMBED_Open_Data/blob/main/DCM_to_PNG.ipynb
# Get DICOM image metadata
class DCM_Tags:
    def __init__(self, img_dcm):
        try:
            self.laterality = img_dcm.ImageLaterality
        except AttributeError:
            self.laterality = np.nan

        try:
            self.view = img_dcm.ViewPosition
        except AttributeError:
            self.view = np.nan

        try:
            self.orientation = img_dcm.PatientOrientation
        except AttributeError:
            self.orientation = np.nan


# Check whether DICOM should be flipped
def check_dcm(imgdcm):
    # Get DICOM metadata
    tags = DCM_Tags(imgdcm)

    # If image orientation tag is defined
    if ~pd.isnull(tags.orientation):
        # CC view
        if tags.view == "CC":
            if tags.orientation[0] == "P":
                flipHorz = True
            else:
                flipHorz = False

            if (tags.laterality == "L") & (tags.orientation[1] == "L"):
                flipVert = True
            elif (tags.laterality == "R") & (tags.orientation[1] == "R"):
                flipVert = True
            else:
                flipVert = False

        # MLO or ML views
        elif (tags.view == "MLO") | (tags.view == "ML"):
            if tags.orientation[0] == "P":
                flipHorz = True
            else:
                flipHorz = False

            if (tags.laterality == "L") & (
                (tags.orientation[1] == "H") | (tags.orientation[1] == "HL")
            ):
                flipVert = True
            elif (tags.laterality == "R") & (
                (tags.orientation[1] == "H") | (tags.orientation[1] == "HR")
            ):
                flipVert = True
            else:
                flipVert = False

        # Unrecognized view
        else:
            flipHorz = False
            flipVert = False

    # If image orientation tag is undefined
    else:
        # Flip RCC, RML, and RMLO images
        if (tags.laterality == "R") & (
            (tags.view == "CC") | (tags.view == "ML") | (tags.view == "MLO")
        ):
            flipHorz = True
            flipVert = False
        else:
            flipHorz = False
            flipVert = False

    return flipHorz, flipVert


def process_image(idx):
    in_path = "/".join(data_df.loc[idx, "anon_dicom_path"].split("/")[5:])
    img_path = data_dir / in_path
    out_img_name = in_path.split("/")[-1][:-4] + ".png"
    out_path = out_dir / out_img_name
    success = False
    if not out_path.exists() and img_path.exists():
        try:
            # img metadata
            window, level, manf = 0, 0, ""
            window = np.array(ast.literal_eval(data_df.loc[idx, "WindowWidth"]))
            level = np.array(ast.literal_eval(data_df.loc[idx, "WindowCenter"]))
            manf = data_df.loc[idx, "Manufacturer"]
            if manf == "GE MEDICAL SYSTEMS" or manf == "GE HEALTHCARE":
                window = window[0]
                level = level[0]
            window, level = float(window), float(level)
            # load image
            dcm = dcmread(img_path)
            flip_h, _ = check_dcm(dcm)
            # resize image
            img = dcm.pixel_array
            if flip_h:
                img = np.fliplr(img)
            img = resize(
                img, output_shape=(IMG_SIZE, IMG_SIZE), preserve_range=True
            ).astype(np.float32)
            # Normalize pixel intensities, and convert to 8-bit
            img -= level - window / 2
            img /= window
            img[img < 0] = 0
            img[img > 1] = 1
            img *= 255
            # save resized image
            imsave(out_path, img.astype(np.uint8))
            success = True
        except Exception as e:
            print(e)
            print(
                f"Couldn't convert image {idx} from patient {data_df.empi_anon[idx]}, Manufacturer: {manf}, Window: {window}, Level: {level}, img_path: {img_path}"
            )
    elif not img_path.exists():
        print(f"Original Image file does not exist at: {img_path} .")
    elif out_path.exists():
        #print(f"Image file {out_path} already exists.")
        success = True
    return success


def main():
    if "preprocessed_img_exists" in data_df.columns:
        to_process = data_df.index[data_df["preprocessed_img_exists"] == False].tolist()
        print(f"... found some already pre-processed images. N={len(to_process)} images left to complete.")
    else:
        to_process = list(range(len(data_df)))
    with mp.Pool(16) as pool:
        successfully_processed = list(
            tqdm(
                pool.imap(process_image, to_process, chunksize=16),
                desc="Processing",
                total=len(to_process),
            )
        )
    data_df["preprocessed_img_exists"] = successfully_processed
    data_df.to_csv(
        "/home/moritz/repositories/fair_leak/data/csv/embed_all.csv", index=False
    )


if __name__ == "__main__":
    main()
