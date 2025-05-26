

import cv2
import numpy as np
import json
import matplotlib.pyplot as plt
import pandas as pd
from sympy.abc import alpha
from scipy.spatial import cKDTree
from src.models.moilutils.moildev import Moildev
from moil_3d_rp import Moil3dAlgorithm



# --- INPUT ---
# pattern_size = (5, 5)  # Ubah sesuai checkerboard kamu
# zoom = 2
#
# # --- LOAD IMAGE ---
# image_original = cv2.imread(image_path)
# image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)
# image_fisheye_with_corners = image_original.copy()
#
# # --- SETUP MOILDEV ---
# moildev = Moildev.Moildev(param)
#
# # --- SIMPAN HASIL KOORDINAT KE CSV ---
# all_fisheye_corners = []

camera_name = "wxsj"

# --- REMAP HELPER UNTUK BALIK KE FISHEYE ---
def remap_to_fisheye(corner_points, mapX, mapY):
    fisheye_points = []
    h, w = mapX.shape
    for pt in corner_points:
        x_remap, y_remap = pt.ravel()
        x_idx = int(round(x_remap))
        y_idx = int(round(y_remap))
        if 0 <= x_idx < w and 0 <= y_idx < h:
            x_fish = mapX[y_idx, x_idx]
            y_fish = mapY[y_idx, x_idx]
            fisheye_points.append((int(x_fish), int(y_fish)))
    return fisheye_points

# --- DISPLAY DAN DETEKSI ---
def proces_image(image_path, param_path, output_prefix, direction_settings=None):
    image_original = cv2.imread(image_path)
    image_gray = cv2.cvtColor(image_original, cv2.COLOR_BGR2GRAY)

    image_fisheye_with_corners = image_original.copy()

    moildev = Moildev.Moildev(param_path)
    # Moildev.Moildev.icy
    # Moildev.Moildev.icx

    all_fisheye_corners = []

    # directions = {
    #     "north": (-96, 6, 7),
    #     "south": (90, 1, 5),
    #     "west": (-1.5, -78, 5),
    #     "east": (2, 94, 4),
    #     "center": (-8, 5.5, 7.5),
    #     "fisheye": None
    # }

    view_images = []

    # for direction, angles in directions.items():
    #     if angles is None:
    #         img_corrected = image_gray.copy()
    #         preview = cv2.resize(img_corrected, (400, 400))

    for direction, setting in direction_settings.items():
        if setting is None:
            img_corrected = image_gray.copy()
            img_detected = img_corrected.copy()
            preview = cv2.resize(img_detected, (400, 400))
        else:
            angles = setting.get("angles")
            pattern_size = setting.get("pattern_size")
            pitch, yaw, zoom = angles

            maps_x, maps_y = moildev.maps_anypoint_mode2(pitch, yaw, zoom)
            # moildev.get_alpha_beta()
            maps_x = maps_x.copy()
            maps_y = maps_y.copy()
            np.save(f'{output_prefix}_maps_x_{direction}_{camera_name}.npy', maps_x)
            np.save(f'{output_prefix}_maps_y_{direction}_{camera_name}.npy', maps_y)

            h_map, w_map = maps_x.shape
            image_resized = cv2.resize(image_gray, (w_map, h_map))

            img_corrected = cv2.remap(
                image_resized,
                maps_x,
                maps_y,
                interpolation=cv2.INTER_CUBIC
            )


            # --- DETEKSI CHECKERBOARD ---
            # ret, corners = cv2.findChessboardCornersSB(
            #     img_corrected,
            #     pattern_size,
            #     flags=cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_EXHAUSTIVE
            # )
            img_med = cv2.medianBlur(img_corrected, 5)  # kernel 5×5

            ret_thresh, img_thresh = cv2.threshold(img_med, 60, 255, cv2.THRESH_BINARY +
                                            cv2.THRESH_OTSU)


            flags = (
                    cv2.CALIB_CB_ADAPTIVE_THRESH |
                    cv2.CALIB_CB_NORMALIZE_IMAGE |
                    cv2.CALIB_CB_FILTER_QUADS
            )

            ret, corners = cv2.findChessboardCornersSB(
                img_thresh,
                pattern_size,
                flags
            )

            if ret:
                img_detected = cv2.drawChessboardCorners(
                    img_med.copy(), pattern_size, corners, ret
                )
                cv2.imwrite(f"{output_prefix}_output_{direction}_detected_{camera_name}.png", img_detected)

                # Mapping balik titik ke gambar fisheye
                fisheye_points = remap_to_fisheye(corners, maps_x, maps_y)
                for pt in fisheye_points:
                    cv2.circle(image_fisheye_with_corners, pt, 5, (0, 255, 0), -1)

                # Simpan ke list global
                for idx, pt in enumerate(fisheye_points):
                    all_fisheye_corners.append({
                        "direction": direction,
                        "point_id": idx,
                        "x": pt[0],
                        "y": pt[1]
                    })
            else:
                img_detected = img_med

            # Simpan hasil dewarp
            cv2.imwrite(f"output_{direction}_{output_prefix}_{camera_name}.png", img_corrected)

            # Siapkan preview untuk grid
            preview = cv2.resize(img_detected, (400, 400))

        # Tambahkan label arah
        preview = cv2.putText(preview, direction.upper(), (10, 30),
                              cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        view_images.append(preview)

    # Lengkapi grid menjadi 2x3
    while len(view_images) < 6:
        empty = np.zeros_like(view_images[0])
        view_images.append(empty)

    row1 = cv2.hconcat(view_images[:3])
    row2 = cv2.hconcat(view_images[3:])
    img_grid = cv2.vconcat([row1, row2])

    cv2.imshow(f"All Compass Views + Checkerboard Detection({output_prefix})", img_grid)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    for row in all_fisheye_corners:
        x, y = int(row["x"]), int(row["y"])
        label = str(row["point_id"])
        cv2.putText(image_fisheye_with_corners, label, (x + 5, y - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1, cv2.LINE_AA)

        alpha, beta = moildev.get_alpha_beta(x, y)
        row["alpha"] = alpha
        row["beta"] = beta

        # Moil3dAlgorithm.nearest_mid_3d_coord()
    # Simpan fisheye dengan corner
    cv2.imwrite(f"fisheye_with_corners_{output_prefix}_{camera_name}.png", image_fisheye_with_corners)


    # Simpan semua titik checkerboard ke CSV
    df = pd.DataFrame(all_fisheye_corners)
    df.to_csv(f"corners_fisheye_{output_prefix}_{camera_name}.csv", index=False)
    return df

# def compute_3d_points(df_l, df_r, cam_l, cam_r):
#     point_3d = []
#     for _, row_l in df_l.iterrows():
#         direction = row_l["direction"]
#         point_id = row_l["point_id"]
#
#         match = df_r[(df_r["direction"] == direction) & (df_r["point_id"] == point_id)]
#         if match.empty:
#             continue
#
#         row_r = match.iloc[0]
#         alpha_l, beta_l = row_l["alpha"], row_l["beta"]
#         alpha_r, beta_r = row_r["alpha"], row_r["beta"]
#
#         if None in [alpha_l, beta_l, alpha_r, beta_r]:
#             continue
#
#         point3d = Moil3dAlgorithm.nearest_mid_3d_coord(
#             alpha_l, beta_l,
#             alpha_r, beta_r,
#             cam_l, cam_r
#         )
#
#         point_3d.append({
#             "direction": direction,
#             "point_id": point_id,
#             "x_3d": point3d[0],
#             "y_3d": point3d[1],
#             "z_3d": point3d[2]
#         })
#
#         return pd.DataFrame(point_3d)

def match_nearest(df_L, df_R, max_dist=200):
    matched = []
    grouped_L = df_L.groupby("direction")
    grouped_R = df_R.groupby("direction")
    for d in set(grouped_L.groups.keys()) & set(grouped_R.groups.keys()):
        pts_L = grouped_L.get_group(d)[["x", "y"]].values
        pts_R = grouped_R.get_group(d)[["x", "y"]].values
        tree_R = cKDTree(pts_R)

        for idx_L, (x_L, y_L) in enumerate(pts_L):
            dist, idx_R = tree_R.query([x_L, y_L], distance_upper_bound=max_dist)
            if dist != float("inf"):
                matched.append({
                    "direction": d,
                    "x_L": x_L, "y_L": y_L, "idx_L": idx_L,
                    "x_R": pts_R[idx_R][0], "y_R": pts_R[idx_R][1], "idx_R": idx_R
                })
    return pd.DataFrame(matched)

def compute_3d_points_from_matches(matched_df, df_L, df_R, cam_L, cam_R):
    points_3d = []
    for _, row in matched_df.iterrows():
        direction = row["direction"]
        idx_L = row["idx_L"]
        idx_R = row["idx_R"]

        row_L = df_L[(df_L["direction"] == direction)].iloc[idx_L]
        row_R = df_R[(df_R["direction"] == direction)].iloc[idx_R]

        alpha_L, beta_L = row_L["alpha"], row_L["beta"]
        alpha_R, beta_R = row_R["alpha"], row_R["beta"]

        if None in [alpha_L, beta_L, alpha_R, beta_R]:
            continue

        point_p, point_q, point3d, ray_angle_deg, confidence = Moil3dAlgorithm.nearest_2_view_points_mid_3d(
            alpha_L, beta_L,
            alpha_R, beta_R,
            cam_L, cam_R
        )

        points_3d.append({
            "direction": direction,
            "idx_L": int(idx_L),
            "idx_R": int(idx_R),
            "point_p": [float(x) for x in point_p],
            "point_q": [float(x) for x in point_q],
            "x": float(point3d[0]),
            "y": float(point3d[1]),
            "z": float(point3d[2]),
            "ray_angle_deg": float(ray_angle_deg),
            "confidence": float(confidence)
        })
    return pd.DataFrame(points_3d)


def compute_3d_points(df_L, df_R, cam_L, cam_R):
    points_3d = []
    for _, row_L in df_L.iterrows():
        direction = row_L["direction"]
        point_id = row_L["point_id"]

        match = df_R[(df_R["direction"] == direction) & (df_R["point_id"] == point_id)]
        if match.empty:
            continue

        row_R = match.iloc[0]
        alpha_L, beta_L = row_L["alpha"], row_L["beta"]
        alpha_R, beta_R = row_R["alpha"], row_R["beta"]

        if None in [alpha_L, beta_L, alpha_R, beta_R]:
            continue

        point_p, point_q, point3d, ray_angle_deg, confidence = Moil3dAlgorithm.nearest_2_view_points_mid_3d(
            alpha_L, beta_L,
            alpha_R, beta_R,
            cam_L, cam_R
        )

        points_3d.append({
            # "direction": direction,
            # "point_id": point_id,
            # "x_3d": point3d[0],
            # "y_3d": point3d[1],
            # "z_3d": point3d[2]
            "direction": direction,
            "point_id": point_id,
            "point_p": [float(point_p[0]), float(point_p[1]), float(point_p[2])],
            "point_q": [float(point_q[0]), float(point_q[1]), float(point_q[2])],
            "x": float(point3d[0]),
            "y": float(point3d[1]),
            "z": float(point3d[2]),
            "ray_angle_deg": float(ray_angle_deg),
            "confidence": float(confidence)
        })
    return pd.DataFrame(points_3d)



# --- MAIN ---
if __name__ == "__main__":

    direction_settings_L = {
        "north": {"angles": (-78, 0, 4.4), "pattern_size": (11, 8)},  # bisa 11,10
        "south": {"angles": (90, 1, 3), "pattern_size": (11, 8)},  # tidak ada gambar
        "west": {"angles": (0, -84.2, 3.3), "pattern_size": (11, 8)},
        # "east": {"angles": (-1, 91, 6), "pattern_size": (11, 11)},#untuk kamera L
        "east": {"angles": (0, 90, 5), "pattern_size": (11, 8)},  # UNTUK CAMERA R
        "center": {"angles": (0.5, 0, 4), "pattern_size": (11, 11)},
        "fisheye": None
    }

    direction_settings_R = {
        "north": {"angles": (-80.30, 0, 4.2), "pattern_size": (11, 8)},
        "south": {"angles": (90, 1, 3), "pattern_size": (11, 8)},  # tidak ada gambar
        "west": {"angles": (0, -90, 5), "pattern_size": (11, 9)},
        "east": {"angles": (0, 86.80, 2.65), "pattern_size": (11, 8)},
        "center": {"angles": (0, 0, 4), "pattern_size": (11, 11)},
        "fisheye": None
    }

    param_R = "D:/hafiqi_work/work/moilapp_perseverance/test/detect chesboard/detection and 3d/wxsj_image/wxsj_7730_2.json"
    param_L = "D:/hafiqi_work/work/moilapp_perseverance/test/detect chesboard/detection and 3d/wxsj_image/wxsj_7730_2.json"
    df_R = proces_image("D:/hafiqi_work/work/moilapp_perseverance/test/detect chesboard/detection and 3d/wxsj_image/wxsj_2_right.png", param_R,"R", direction_settings=direction_settings_R)
    df_L = proces_image("D:/hafiqi_work/work/moilapp_perseverance/test/detect chesboard/detection and 3d/wxsj_image/wxsj_2_left.png", param_L,"L", direction_settings=direction_settings_L)

    cam_R = (50, 0, 0)
    cam_L = (-50, 0, 0)  # baseline 10 cm

    matched_df = match_nearest(df_L, df_R, max_dist=200)

    print("✔️ The results found in matching:")
    print(matched_df['direction'].value_counts())

    df_3d = compute_3d_points_from_matches(matched_df, df_L, df_R, cam_L, cam_R)

    # df_3d = compute_3d_points(df_L, df_R, cam_L, cam_R)
    # df_3d.to_csv("triangulated_points_3d.csv", index=False)

    # Save to JSON in x, z, y order
    json_data = []
    for idx, row in df_3d.iterrows():
        json_data.append({
            "direction": row["direction"],
            "point_id": int(idx),
            "idx_L ": int(row["idx_L"]),
            "idx_R ": int(row["idx_R"]),
            "point_p": row["point_p"],
            "point_q": row["point_q"],
            "x": float(row["x"]),
            "y": float(row["y"]),
            "z": float(row["z"]),
            "ray_angle_deg": float(row["ray_angle_deg"]),
            "confidence": float(row["confidence"])
        })



    with open(f"triangulated_points_3d_{camera_name}.json", "w") as f:
        json.dump(json_data, f, indent=2)

    matched_df.to_csv("matched_pairs.csv", index=False)



