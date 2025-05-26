
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from collections import defaultdict
import plotly.graph_objects as go
from scipy.optimize import least_squares
from moil_3d_rp import Reprojector_3d


camera_name = "wxsj"
# -----------------------------------------------------------------------------
# 1) Hitung sudut antar dua normal
# -----------------------------------------------------------------------------
def plane_angle(A1,B1,C1, A2,B2,C2):
    dot = A1*A2 + B1*B2 + C1*C2
    n1 = np.linalg.norm([A1,B1,C1])
    n2 = np.linalg.norm([A2,B2,C2])
    return np.degrees(np.arccos(np.clip(dot/(n1*n2), -1,1)))

# -----------------------------------------------------------------------------
# 2) Orthogonal snap ke plane
# -----------------------------------------------------------------------------
def snap_to_plane(grouped, plane_normals, plane_D):
    out = {}
    for d,pts in grouped.items():
        A,B,C = plane_normals[d]; D = plane_D[d]
        norm = np.linalg.norm([A,B,C])
        snapped = []
        for p in pts:
            x,y,z = p['x'],p['y'],p['z']
            d_signed = (A*x + B*y + C*z + D)/norm
            x2 = x - d_signed*(A/norm)
            y2 = y - d_signed*(B/norm)
            z2 = z - d_signed*(C/norm)
            q = p.copy(); q.update({'x':x2,'y':y2,'z':z2})
            snapped.append(q)
        out[d] = snapped
    return out

# -----------------------------------------------------------------------------
# 3) Non-linear per-point refine
# -----------------------------------------------------------------------------
# def refine_per_point(grouped_snapped,
#                      dfL, dfR,
#                      rp_L,rp_R, cam_L, cam_R,
#                      plane_normals, plane_D,
#                      plane_weight=0.5):
#     """
#     grouped_snapped: dict[direction] -> list of snapped 3D points
#     dfL, dfR: DataFrame with columns ['direction','point_id','x','y']
#     rp: your Reprojector instance
#     cam_L, cam_R: 3‐tuples of camera centers
#     plane_normals, plane_D: dicts giving plane equation for each direction
#     plane_weight: λ in the residual
#     """
#     refined = {}
#     for d, pts in grouped_snapped.items():
#         A, B, C = plane_normals[d]
#         D       = plane_D[d]
#
#         refined_pts = []
#         for p in pts:
#             pid  = p['point_id']
#             rowL = dfL.query("direction==@d and point_id==@pid").iloc[0]
#             rowR = dfR.query("direction==@d and point_id==@pid").iloc[0]
#
#             def resid(X):
#                 # — reprojection into left camera —
#                 p_camL       = (X[0]-cam_L[0], X[1]-cam_L[1], X[2]-cam_L[2])
#                 alpha_L,β_L  = rp_L._pt3d_to_alphabeta(p_camL)
#                 uL, vL       = rp_L._alphabeta_to_pixel(alpha_L, β_L)
#
#                 # — reprojection into right camera —
#                 p_camR       = (X[0]-cam_R[0], X[1]-cam_R[1], X[2]-cam_R[2])
#                 alpha_R,β_R  = rp_R._pt3d_to_alphabeta(p_camR)
#                 uR, vR       = rp_R._alphabeta_to_pixel(alpha_R, β_R)
#
#                 # 2D reprojection residuals
#                 reproj = [
#                     uL - rowL.x,  vL - rowL.y,
#                     uR - rowR.x,  vR - rowR.y
#                 ]
#
#                 # plane‐distance residual (signed distance)
#                 plane_dist = (A*X[0] + B*X[1] + C*X[2] + D) \
#                              / np.sqrt(A*A + B*B + C*C)
#
#                 return reproj + [plane_weight * plane_dist]
#
#             # initialize at the snapped point
#             X0  = np.array([p['x'], p['y'], p['z']])
#             sol = least_squares(resid, X0, verbose=0)
#
#             q = p.copy()
#             q.update(x=sol.x[0], y=sol.x[1], z=sol.x[2])
#             refined_pts.append(q)
#
#         refined[d] = refined_pts
#
#     return refined

def refine_per_point_from_matches(pts_3d, dfL, dfR, rp_L, rp_R, cam_L, cam_R, plane_normals, plane_D, plane_weight=0.5):
    grouped = defaultdict(list)
    for p in pts_3d:
        d = p['direction']
        # idx_L = p['idx_L']
        # idx_R = p['idx_R']
        pid = p['point_id']  # pakai point_id dari JSON

        # Cek apakah index valid

        rowL = dfL.query("direction == @d and point_id == @pid")
        rowR = dfR.query("direction == @d and point_id == @pid")
        if rowL.empty or rowR.empty:
            continue

        rowL = rowL.iloc[0]
        rowR = rowR.iloc[0]

        A, B, C = plane_normals[d]
        D = plane_D[d]

        def resid(X):
            p_camL = (X[0]-cam_L[0], X[1]-cam_L[1], X[2]-cam_L[2])
            alpha_L, beta_L = rp_L._pt3d_to_alphabeta(p_camL)
            uL, vL = rp_L._alphabeta_to_pixel(alpha_L, beta_L)

            p_camR = (X[0]-cam_R[0], X[1]-cam_R[1], X[2]-cam_R[2])
            alpha_R, beta_R = rp_R._pt3d_to_alphabeta(p_camR)
            uR, vR = rp_R._alphabeta_to_pixel(alpha_R, beta_R)

            reproj = [uL - rowL.x, vL - rowL.y, uR - rowR.x, vR - rowR.y]
            plane_dist = (A*X[0] + B*X[1] + C*X[2] + D) / np.sqrt(A*A + B*B + C*C)
            return reproj + [plane_weight * plane_dist]

        X0 = np.array([p['x'], p['y'], p['z']])
        sol = least_squares(resid, X0, verbose=0)
        p_new = p.copy()
        p_new.update(x=sol.x[0], y=sol.x[1], z=sol.x[2])
        grouped[d].append(p_new)

    return grouped


def find_4_corners_directional(pts, direction):
    """

    Args: find 4 corner for all of direction
        pts:
        direction:

    Returns:

    """
    axes_map = {
        'north': ('x', 'z'),
        'south': ('x', 'z'),
        'east':  ('y', 'z'),
        'west':  ('y', 'z'),
        'top':   ('x', 'y'),
        'bottom':('x', 'y'),
        'center':('x', 'y')
    }
    key1, key2 = axes_map.get(direction, ('x', 'y'))

    arr = np.array([[p[key1], p[key2]] for p in pts])
    min1, max1 = arr[:, 0].min(), arr[:, 0].max()
    min2, max2 = arr[:, 1].min(), arr[:, 1].max()

    def closest(pt_list, v1, v2):
        return min(pt_list, key=lambda p: (p[key1] - v1)**2 + (p[key2] - v2)**2)

    return [
        closest(pts, min1, min2),
        closest(pts, min1, max2),
        closest(pts, max1, min2),
        closest(pts, max1, max2),
    ]
# -----------------------------------------------------------------------------
# 4) Simpan interactive HTML
# -----------------------------------------------------------------------------
def save_html_planes(grouped, raw_coeffs, annotation, path_html):
    dirs_y,dirs_x,dirs_z = ['north'], ['west','east'], ['center','top']
    # corners=[0,10,110,120]
    colors = {'north':'cyan','west':'orange','east':'orange',
              'center':'gray','top':'gray','south':'cyan'}
    fig=go.Figure()
    allx,ally,allz = [],[],[]
    # scatter
    for d,pts in grouped.items():
        X=[p['x'] for p in pts]
        Y=[p['y'] for p in pts]
        Z=[p['z'] for p in pts]
        allx+=X; ally+=Y; allz+=Z
        fig.add_trace(go.Scatter3d(x=X,y=Y,z=Z,mode='markers',
            marker=dict(size=3,color=colors.get(d,'black')),name=f"{d}"))
    r = max(np.ptp(allx), np.ptp(ally), np.ptp(allz))/2
    # axes
    fig.add_trace(go.Scatter3d(x=[-r,r],y=[0,0],z=[0,0],mode='lines',line=dict(color='red',width=4)))
    fig.add_trace(go.Scatter3d(x=[0,0],y=[-r,r],z=[0,0],mode='lines',line=dict(color='green',width=4)))
    fig.add_trace(go.Scatter3d(x=[0,0],y=[0,0],z=[-r,r],mode='lines',line=dict(color='black',width=4)))
    # cameras
    fig.add_trace(go.Scatter3d(x=[cam_R[0]],y=[0],z=[0],mode='markers',marker=dict(size=6,color='yellow'),name='Cam R'))
    fig.add_trace(go.Scatter3d(x=[cam_L[0]],y=[0],z=[0],mode='markers',marker=dict(size=6,color='magenta'),name='Cam L'))
    # quads
    for d,pts in grouped.items():
        a,b,c0 = raw_coeffs[d]
        # p0,p1,p2,p3 = [pts[i] for i in corners]
        p0, p1, p2, p3 = find_4_corners_directional(pts, d)
        if d in dirs_y:
            X=[[p0['x'],p1['x']],[p2['x'],p3['x']]]
            Z=[[p0['z'],p1['z']],[p2['z'],p3['z']]]
            Y=[[a*X[i][j]+b*Z[i][j]+c0 for j in (0,1)] for i in (0,1)]
        elif d in dirs_x:
            Y=[[p0['y'],p1['y']],[p2['y'],p3['y']]]
            Z=[[p0['z'],p1['z']],[p2['z'],p3['z']]]
            X=[[a*Y[i][j]+b*Z[i][j]+c0 for j in (0,1)] for i in (0,1)]
        else:
            X=[[p0['x'],p1['x']],[p2['x'],p3['x']]]
            Y=[[p0['y'],p1['y']],[p2['y'],p3['y']]]
            Z=[[a*X[i][j]+b*Y[i][j]+c0 for j in (0,1)] for i in (0,1)]
        fig.add_trace(go.Surface(x=X,y=Y,z=Z,opacity=0.5,showscale=False,
            surfacecolor=[[0,0],[0,0]],colorscale=[[0,colors[d]],[1,colors[d]]]))
    fig.update_layout(
        annotations=[dict(text=annotation,showarrow=False,
            x=0,y=1.05,xref='paper',yref='paper',align='left')],
        scene=dict(aspectmode='cube'),
        margin=dict(l=0,r=0,b=0,t=50)
    )
    fig.write_html(path_html)
    print("✅", path_html)

# -----------------------------------------------------------------------------
# 5) Hitung reproj-error 2D & simpan CSV
# -----------------------------------------------------------------------------
# def compute_errors(snapped_json, out_csv, df_det, cam):
#     pts = json.load(open(snapped_json))
#     rows=[]
#     for r in pts:
#         u,v = rp.alphabeta_to_pixel(*rp.pt3d_to_alphabeta(
#             (r['x']-cam[0], r['y']-cam[1], r['z']-cam[2])
#         ))
#         rows.append((r['direction'],r['point_id'],int(u),int(v)))
#     df_RE = pd.DataFrame(rows,columns=['direction','point_id','u','v'])
#     df = df_RE.merge(df_det, on=['direction','point_id'])
#     df['err']=np.hypot(df.u-df.x, df.v-df.y)
#
#     # Hitung mean & rms
#     mean, rms = df['err'].mean(), np.sqrt((df['err'] ** 2).mean())
#     # Buat baris summary
#     summary = pd.DataFrame([
#         {'direction': '', 'point_id': '', 'u': 'Mean err', 'v': '', 'x': '', 'y': '', 'err': mean},
#         {'direction': '', 'point_id': '', 'u': 'RMS err', 'v': '', 'x': '', 'y': '', 'err': rms},
#     ])
#
#     # Gabungkan
#     df_out = pd.concat([df, summary], ignore_index=True)
#
#     # Simpan
#     df_out.to_csv(out_csv, index=False)
#     print(f"✅ {out_csv}: mean={mean:.2f}, rms={rms:.2f}")

def compute_errors(snapped_json, out_csv, df_det, cam, rp_L, rp_R, side='left'):
    pts = json.load(open(snapped_json))
    rows = []
    missing = 0
    for r in pts:

        dir = r['direction']
        pid = r['point_id']

        pt_world = np.array([r['x'], r['y'], r['z']], dtype=float)

        if side == 'left':
            u, v = rp_L._reproject_point(pt_world)
        else:
            u, v = rp_R._reproject_point(pt_world)

        # Cari titik deteksi asli
        row_det = df_det[(df_det['direction'] == dir) & (df_det['point_id'] == pid)]
        if row_det.empty:
            missing += 1
            continue

        x = row_det.iloc[0]['x']
        y = row_det.iloc[0]['y']
        err = np.hypot(u - x, v - y)

        # Tampilkan peringatan untuk error besar
        if err > 100:
            print(
                f"[WARNING {side.upper()}] ID={pid}, Dir={dir} | Det=({x:.1f},{y:.1f}) → Reproj=({u:.1f},{v:.1f}) | ERR={err:.1f}")

        # print(f"[{side.upper()}] Point ID: {r['point_id']}, Direction: {r['direction']}")
        # print(f"  3D World Point : {pt_world}")
        # print(f"  Cam Coord      : {cam}")
        # print(f"  Transformed 3D : {pt_cam}")
        # print(f"  Alpha, Beta    : {alpha:.2f}, {beta:.2f}")
        # print(f"  Original point : ({x:.1f}, {y:.1f})")
        # print(f"  Projected Pixel: ({u:.1f}, {v:.1f})")
        # print("-" * 40)


        # Simpan semua data
        rows.append((dir, pid, int(round(u)), int(round(v)), float(x), float(y), float(err)))


        # u,v = rp._alphabeta_to_pixel(*rp._pt3d_to_alphabeta(
        #     (r['x']-cam[0], r['y']-cam[1], r['z']-cam[2])
        # ))
        # rows.append((r['direction'], r['point_id'], int(u), int(v)))
    # df_RE = pd.DataFrame(rows, columns=['direction','point_id','u','v'])
    # df = df_RE.merge(df_det, on=['direction','point_id'])
    # df['err'] = np.hypot(df.u - df.x, df.v - df.y)
    # df.to_csv(out_csv, index=False)

    # Konversi ke DataFrame dan simpan
    df_RE = pd.DataFrame(rows, columns=['direction', 'point_id', 'u', 'v', 'x', 'y', 'err'])
    df_RE.to_csv(out_csv, index=False)
    # df_RE.to_csv(out_csv, index=False)
    # mean, rms = df.err.mean(), np.sqrt((df.err**2).mean())

    mean = df_RE['err'].mean()
    rms = np.sqrt((df_RE['err'] ** 2).mean())
    print(f"✅ {out_csv}: mean={mean:.2f}, rms={rms:.2f}")

    # mean, rms = df.err.mean(), np.sqrt((df.err**2).mean())
    # df.to_csv(out_csv,index=False)
    # print(f"✅ {out_csv}: mean={mean:.2f}, rms={rms:.2f}")

def compute_errors_from_matches(pts_3d_json, matched_csv, df_det, side, rp_L, rp_R, out_csv):
    pts_3d = json.load(open(pts_3d_json))
    matched_df = pd.read_csv(matched_csv)
    grouped_3d = {(p['direction'], p['idx_L'], p['idx_R']): p for p in pts_3d}

    rows = []
    for _, row in matched_df.iterrows():
        d = row['direction']
        key = (d, row['idx_L'], row['idx_R'])

        if key not in grouped_3d:
            continue

        pt = grouped_3d[key]
        pt3d = [pt['x'], pt['y'], pt['z']]
        u, v = (rp_L if side == 'left' else rp_R)._reproject_point(pt3d)

        x = row['x_L'] if side == 'left' else row['x_R']
        y = row['y_L'] if side == 'left' else row['y_R']
        err = np.hypot(u - x, v - y)

        rows.append((d, row['idx_L'] if side == 'left' else row['idx_R'], int(round(u)), int(round(v)), x, y, err))

    df_err = pd.DataFrame(rows, columns=['direction', 'match_id', 'u', 'v', 'x', 'y', 'err'])
    df_err.to_csv(out_csv, index=False)
    print(f"✅ {out_csv}: mean={df_err['err'].mean():.2f}, rms={np.sqrt((df_err['err']**2).mean()):.2f}")


def debug_reprojection_consistency(pt3d, rp1, rp2):
    a1, b1 = rp1._pt3d_to_alphabeta(pt3d)
    a2, b2 = rp2._pt3d_to_alphabeta(pt3d)
    u1, v1 = rp1._alphabeta_to_pixel(a1, b1)
    u2, v2 = rp2._alphabeta_to_pixel(a2, b2)
    print("=== Reprojection Comparison ===")
    print(f"3D point: {pt3d}")
    print(f"[RP1] alpha={a1:.2f}, beta={b1:.2f}, u={u1:.2f}, v={v1:.2f}")
    print(f"[RP2] alpha={a2:.2f}, beta={b2:.2f}, u={u2:.2f}, v={v2:.2f}")
    print(f"Δalpha={abs(a1-a2):.3f}, Δbeta={abs(b1-b2):.3f}")
    print(f"Δu={abs(u1-u2):.2f}, Δv={abs(v1-v2):.2f}")
    print("="*40)

# -----------------------------------------------------------------------------
# 6) Overlay 2D sebelum/snap/refine
# -----------------------------------------------------------------------------
# def overlay_2d(img, df_det,csv_orig, csv_snap, csv_ref, side):
#     # load original reprojections
#     df_o = (
#         pd.read_csv(csv_orig)
#         .merge(df_det, on=['direction', 'point_id'])
#         .rename(columns={'u': 'u_o', 'v': 'v_o'})
#     )
#     df_s = pd.read_csv(csv_snap).merge(df_det,on=['direction','point_id']).rename(columns={'u':'u_s','v':'v_s'})
#     df_r = pd.read_csv(csv_ref).merge(df_det,on=['direction','point_id']).rename(columns={'u':'u_r','v':'v_r'})
#
#     # detected vs ori
#     plt.figure(figsize=(6, 6))
#     plt.imshow(img, origin='upper')
#     plt.scatter(df_det.x, df_det.y, c='lime', s=15, label='detected')
#     plt.scatter(df_o.u_o, df_o.v_o, c='yellow', marker='+', s=40, label='snapped')
#     plt.axis('off');plt.legend(loc='upper right')
#     plt.savefig(f"overlay_{side}_ori.png", dpi=200)
#     # before vs snap
#     plt.figure(figsize=(6,6))
#     plt.imshow(img,origin='upper')
#     plt.scatter(df_det.x,df_det.y,c='lime',s=15,label='detected')
#     plt.scatter(df_s.u_s,df_s.v_s,c='red',marker='+',s=40,label='snapped')
#     plt.axis('off'); plt.legend(loc='upper right')
#     plt.savefig(f"overlay_{side}_snap.png",dpi=200)
#     # before vs refine
#     plt.figure(figsize=(6,6))
#     plt.imshow(img,origin='upper')
#     plt.scatter(df_det.x,df_det.y,c='lime',s=15,label='detected')
#     plt.scatter(df_r.u_r,df_r.v_r,c='blue',marker='x',s=40,label='refined')
#     plt.axis('off'); plt.legend(loc='upper right')
#     plt.savefig(f"overlay_{side}_refine.png",dpi=200)
#
#     print("✅ save image ")

def overlay_2d_match(img, df_det, csv_orig, csv_snap, csv_ref, side):
    df_o = pd.read_csv(csv_orig).rename(columns={'u': 'u_o', 'v': 'v_o'})
    df_s = pd.read_csv(csv_snap).rename(columns={'u': 'u_s', 'v': 'v_s'})
    df_r = pd.read_csv(csv_ref).rename(columns={'u': 'u_r', 'v': 'v_r'})

    # Detected point positions pakai match_id
    plt.figure(figsize=(10, 10))
    plt.imshow(img, origin='upper')
    plt.scatter(df_o.x, df_o.y, c='lime', s=15, label='detected')
    plt.scatter(df_o.u_o, df_o.v_o, c='yellow', marker='+', s=40, label='original reproj')
    for _, row in df_o.iterrows():
        plt.text(row.x + 5, row.y - 5, str(int(row.match_id)), fontsize=6, color='lime')
        plt.text(row.u_o + 5, row.v_o + 5, str(int(row.match_id)), fontsize=6, color='yellow')
    ...



def overlay_2d(img, df_det, csv_orig, csv_snap, csv_ref, side):
    # — load and clean original reprojections —
    df_o = pd.read_csv(csv_orig)
    # drop rows where point_id isn’t a number (i.e. your Mean/RMS summary lines)
    df_o = df_o[pd.to_numeric(df_o['point_id'], errors='coerce').notnull()]\
             .merge(df_det, on=['direction','point_id'])\
             .rename(columns={'u':'u_o','v':'v_o'})

    # — load and clean snapped reprojections —
    df_s = pd.read_csv(csv_snap)
    df_s = df_s[pd.to_numeric(df_s['point_id'], errors='coerce').notnull()]\
             .merge(df_det, on=['direction','point_id'])\
             .rename(columns={'u':'u_s','v':'v_s'})

    # — load and clean refined reprojections —
    df_r = pd.read_csv(csv_ref)
    df_r = df_r[pd.to_numeric(df_r['point_id'], errors='coerce').notnull()]\
             .merge(df_det, on=['direction','point_id'])\
             .rename(columns={'u':'u_r','v':'v_r'})

    # now plot
    plt.figure(figsize=(10,10))
    plt.imshow(img, origin='upper')
    plt.scatter(df_det.x, df_det.y,   c='lime', s=15, label='detected')
    plt.scatter(df_o.u_o, df_o.v_o,   c='yellow', marker='+', s=40, label='original reproj')
    # tambahkan label point_id
    for _, row in df_det.iterrows():
        plt.text(row.x + 5, row.y - 5, str(int(row.point_id)), fontsize=6, color='lime')
    for _, row in df_o.iterrows():
        plt.text(row.u_o + 5, row.v_o + 5, str(int(row.point_id)), fontsize=6, color='yellow')

    plt.axis('off'); plt.legend(loc='upper right')
    plt.savefig(f"overlay_{camera_name}_{side}_orig.png", dpi=200)

    plt.figure(figsize=(10,10))
    plt.imshow(img, origin='upper')
    plt.scatter(df_det.x, df_det.y,   c='lime', s=15, label='detected')
    plt.scatter(df_s.u_s, df_s.v_s,   c='red',    marker='+', s=40, label='snapped')
    for _, row in df_s.iterrows():
        plt.text(row.u_s + 5, row.v_s + 5, str(int(row.point_id)), fontsize=6, color='red')
    for _, row in df_det.iterrows():
        plt.text(row.x + 5, row.y - 5, str(int(row.point_id)), fontsize=6, color='lime')
    plt.axis('off'); plt.legend(loc='upper right')
    plt.savefig(f"overlay_{camera_name}_{side}_snap.png", dpi=200)

    plt.figure(figsize=(10,10))
    plt.imshow(img, origin='upper')
    plt.scatter(df_det.x, df_det.y,   c='lime', s=15, label='detected')
    plt.scatter(df_r.u_r, df_r.v_r,   c='blue',   marker='x', s=40, label='refined')
    for _, row in df_r.iterrows():
        plt.text(row.u_r + 5, row.v_r + 5, str(int(row.point_id)), fontsize=6, color='blue')
    for _, row in df_det.iterrows():
        plt.text(row.x + 5, row.y - 5, str(int(row.point_id)), fontsize=6, color='lime')
    plt.axis('off'); plt.legend(loc='upper right')
    plt.savefig(f"overlay_{camera_name}_{side}_refine.png", dpi=200)

    print("✅ Labeled overlay images saved.")
# -----------------------------------------------------------------------------
# MAIN
# -----------------------------------------------------------------------------

if __name__=="__main__":
    # paths
    js3d = f"triangulated_points_3d_{camera_name}.json"
    param_L = "wxsj_image/wxsj_7730_2.json"
    param_R = "wxsj_image/wxsj_7730_2.json"
    detL_csv = f"corners_fisheye_L_{camera_name}.csv"
    detR_csv = f"corners_fisheye_R_{camera_name}.csv"
    cam_L   = (-50,0,0); cam_R = (50,0,0)

    # load 3D & group
    data = json.load(open(js3d))
    grouped = defaultdict(list)
    for e in data: grouped[e['direction']].append(e)
    for d in grouped: grouped[d].sort(key=lambda r:r['point_id'])

    # fit planes
    dirs_y,dirs_x,dirs_z = ['north'], ['west','east'], ['center','top']
    plane_normals,plane_D,distances,raw_coeffs = {},{},{},{}


    for d,pts in grouped.items():
        M = np.array([[p['x'],p['y'],p['z']] for p in pts])
        x,y,z = M[:,0],M[:,1],M[:,2]
        if d in dirs_y:
            A = np.vstack([x,z,np.ones_like(x)]).T; a,b,c = np.linalg.lstsq(A,y,rcond=None)[0]
            A_c,B_c,C_c,D_c = a,-1,b,c
        elif d in dirs_x:
            A = np.vstack([y,z,np.ones_like(y)]).T; a,b,c = np.linalg.lstsq(A,x,rcond=None)[0]
            A_c,B_c,C_c,D_c = -1,a,b,c
        else:
            A = np.vstack([x,y,np.ones_like(x)]).T; a,b,c = np.linalg.lstsq(A,z,rcond=None)[0]
            A_c,B_c,C_c,D_c = a,b,-1,c
        raw_coeffs[d]=(a,b,c)
        plane_normals[d]=[A_c,B_c,C_c]; plane_D[d]=D_c
        num = np.abs(A_c*x + B_c*y + C_c*z + D_c)
        distances[d] = num/np.linalg.norm([A_c,B_c,C_c])
        print(f"[{d:6s}] mean dist={distances[d].mean():.3f}")

    # annotation teks
    annot = "<br>".join(
        ["<b>Mean dist:</b>"] +
        [f"{d}: {distances[d].mean():.3f}" for d in distances] +
        ["<b>Angles:</b>"] +
        [f"{o}: {plane_angle(*plane_normals['center'], *plane_normals[o]):.1f}°"
         for o in dirs_x+dirs_y]
    )

    # 3D Reprojector
    rp_L = Reprojector_3d(param_L, cam_coord=cam_L)
    rp_R = Reprojector_3d(param_R, cam_coord=cam_R)

    for pt in data:
        if pt["direction"] == "center" and pt["point_id"] == 60:
            pt3d = [pt["x"], pt["y"], pt["z"]]
            print(">> DEBUG untuk LEFT camera:")
            debug_reprojection_consistency(pt3d, rp_L, rp_L)
            print(">> DEBUG untuk RIGHT camera:")
            debug_reprojection_consistency(pt3d, rp_R, rp_R)
            break

    # 1) ORIGINAL HTML
    save_html_planes(grouped, raw_coeffs, annot, f"all_planes_original_{camera_name}.html")

    # 2) SNAP
    grouped_snap = snap_to_plane(grouped, plane_normals, plane_D)

    # hitung distances snapped
    distances_snap = {}
    for d, pts in grouped_snap.items():
        X = np.array([p['x'] for p in pts])
        Y = np.array([p['y'] for p in pts])
        Z = np.array([p['z'] for p in pts])
        A, B, C = plane_normals[d]
        D = plane_D[d]
        dist = np.abs(A * X + B * Y + C * Z + D) / np.linalg.norm([A, B, C])
        distances_snap[d] = dist

    # build annotation untuk snapped
    lines_snap = ["<b>Mean distances (snapped):</b>"]
    for d in sorted(distances_snap):
        lines_snap.append(f"{d}: {distances_snap[d].mean():.3f}")
    lines_snap.append("<br><b>Angles (°) (unchanged):</b>")
    for o in dirs_x + dirs_y:
        ang = plane_angle(*plane_normals['center'], *plane_normals[o])
        lines_snap.append(f"center↔{o}: {ang:.1f}°")
    for o in dirs_x:
        ang = plane_angle(*plane_normals['north'], *plane_normals[o])
        lines_snap.append(f"north↔{o}: {ang:.1f}°")
    annot_snap = "<br>".join(lines_snap)

    save_html_planes(grouped_snap, raw_coeffs, annot_snap, f"all_planes_snapped_{camera_name}.html")
    json.dump([p for pts in grouped_snap.values() for p in pts],
              open(f"triangulated_{camera_name}_points_3d_snapped.json","w"), indent=2)

    # 3) REFINE
    dfL = pd.read_csv(detL_csv).rename(columns={'x':'x','y':'y'})
    dfR = pd.read_csv(detR_csv).rename(columns={'x':'x','y':'y'})

    # grouped_ref = refine_per_point(
    #     grouped_snap,
    #     dfL, dfR,
    #     rp_L, rp_R, cam_L, cam_R,
    #     plane_normals, plane_D,
    #     plane_weight=0.5
    # )

    grouped_ref = refine_per_point_from_matches(
        [p for pts in grouped_snap.values() for p in pts],  # flatten grouped_snap
        dfL, dfR,
        rp_L, rp_R, cam_L, cam_R,
        plane_normals, plane_D,
        plane_weight=0.5
    )

    # hitung distances refined
    distances_ref = {}
    for d, pts in grouped_ref.items():
        X = np.array([p['x'] for p in pts])
        Y = np.array([p['y'] for p in pts])
        Z = np.array([p['z'] for p in pts])
        A, B, C = plane_normals[d]
        D = plane_D[d]
        dist = np.abs(A * X + B * Y + C * Z + D) / np.linalg.norm([A, B, C])
        distances_ref[d] = dist

    # build annotation untuk refined
    lines_ref = ["<b>Mean distances (refined):</b>"]
    for d in sorted(distances_ref):
        lines_ref.append(f"{d}: {distances_ref[d].mean():.3f}")
    lines_ref.append("<br><b>Angles (°) (unchanged):</b>")
    for o in dirs_x + dirs_y:
        ang = plane_angle(*plane_normals['center'], *plane_normals[o])
        lines_ref.append(f"center↔{o}: {ang:.1f}°")
    for o in dirs_x:
        ang = plane_angle(*plane_normals['north'], *plane_normals[o])
        lines_ref.append(f"north↔{o}: {ang:.1f}°")
    annot_ref = "<br>".join(lines_ref)

    save_html_planes(grouped_ref, raw_coeffs, annot_ref, f"all_planes_refined_{camera_name}.html")
    json.dump([p for pts in grouped_ref.values() for p in pts],
              open(f"triangulated_{camera_name}_points_3d_refined.json","w"), indent=2)

    # 2D error CSV

    compute_errors_from_matches(js3d, "matched_pairs.csv", dfL, 'left', rp_L, rp_R,
                                f"errors_orig_left_{camera_name}.csv")
    compute_errors_from_matches(f"triangulated_{camera_name}_points_3d_snapped.json", "matched_pairs.csv", dfL, 'left',
                                rp_L, rp_R, f"errors_snap_left_{camera_name}.csv")
    compute_errors_from_matches(f"triangulated_{camera_name}_points_3d_refined.json", "matched_pairs.csv", dfL, 'left',
                                rp_L, rp_R, f"errors_ref_left_{camera_name}.csv")

    compute_errors_from_matches(js3d, "matched_pairs.csv", dfR, 'right', rp_R, rp_L,
                                f"errors_orig_right_{camera_name}.csv")
    compute_errors_from_matches(f"triangulated_{camera_name}_points_3d_snapped.json", "matched_pairs.csv", dfR, 'right',
                                rp_R, rp_L, f"errors_snap_right_{camera_name}.csv")
    compute_errors_from_matches(f"triangulated_{camera_name}_points_3d_refined.json", "matched_pairs.csv", dfR, 'right',
                                rp_R, rp_L, f"errors_ref_right_{camera_name}.csv")

    # compute_errors_from_matches(js3d, f"errors_orig_left_{camera_name}.csv",dfL, cam_L, rp_L, rp_R,side='left')
    # compute_errors_from_matches(f"triangulated_{camera_name}_points_3d_snapped.json", f"errors_snap_left_{camera_name}.csv", dfL, cam_L, rp_L, rp_R, 'left')
    # compute_errors_from_matches(f"triangulated_{camera_name}_points_3d_refined.json",f"errors_ref_left_{camera_name}.csv", dfL, cam_L, rp_L, rp_R,'left')
    #
    #
    # compute_errors_from_matches(js3d, f"errors_orig_right_{camera_name}.csv", dfR, cam_R, rp_R, rp_L, side='right')
    # compute_errors_from_matches(f"triangulated_{camera_name}_points_3d_snapped.json", f"errors_snap_right_{camera_name}.csv", dfR, cam_R, rp_R, rp_L, 'right')
    # compute_errors_from_matches(f"triangulated_{camera_name}_points_3d_refined.json",f"errors_ref_right_{camera_name}.csv", dfR, cam_R, rp_R, rp_L,'right')

    # 2D overlays
    left_img = np.array(Image.open("wxsj_image/wxsj_2_left.png"))
    right_img = np.array(Image.open("wxsj_image/wxsj_2_right.png"))
    overlay_2d_match(left_img, dfL, f"errors_orig_left_{camera_name}.csv",f"errors_snap_left_{camera_name}.csv", f"errors_ref_left_{camera_name}.csv",   "left")
    overlay_2d_match(right_img,dfR, f"errors_orig_right_{camera_name}.csv",f"errors_snap_right_{camera_name}.csv",f"errors_ref_right_{camera_name}.csv", "right")

