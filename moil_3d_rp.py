import math
from typing import Tuple
from sympy import Symbol, Eq, solve
import numpy as np
from src.models.moilutils.moildev import Moildev


class Moil3dAlgorithm:

    @staticmethod
    def _beta_moil2cartesian(beta_moil: float) -> float:
        beta_cartesian = beta_moil % 360
        beta_cartesian = 90 - beta_cartesian

        if beta_cartesian < 0:
            return beta_cartesian + 360
        else:
            return beta_cartesian

    # @staticmethod
    # def al_ba_2_vector_x(alpha: float, beta: float) -> float:
    #     beta = Moil3dAlgorithm._beta_moil2cartesian(beta)
    #     return math.sin(math.radians(alpha)) * math.cos(math.radians(beta))
    #
    # @staticmethod
    # def al_ba_2_vector_y(alpha: float, beta: float) -> float:
    #     beta = Moil3dAlgorithm._beta_moil2cartesian(beta)
    #     return math.sin(math.radians(alpha)) * math.sin(math.radians(beta))
    #
    # @staticmethod
    # def al_2_vector_z(alpha: float) -> float:
    #     return math.cos(math.radians(alpha))

    @staticmethod
    def al_ba_2_vector(alpha: float, beta: float) -> Tuple[float, float, float]:
        beta = Moil3dAlgorithm._beta_moil2cartesian(beta)
        x = math.sin(math.radians(alpha)) * math.cos(math.radians(beta))
        y = math.sin(math.radians(alpha)) * math.sin(math.radians(beta))
        z = math.cos(math.radians(alpha))
        return x, y, z

    # @staticmethod
    # def al_ba_2_vector(alpha: float, beta: float) -> Tuple[float, float, float]:
    #     vector_x = Moil3dAlgorithm.al_ba_2_vector_x(alpha, beta)
    #     vector_y = Moil3dAlgorithm.al_ba_2_vector_y(alpha, beta)
    #     vector_z = Moil3dAlgorithm.al_2_vector_z(alpha)
    #
    #     return vector_x, vector_y, vector_z


    # NEW METHOD: Compute angle between rays
    @staticmethod
    def angle_between_vectors(v1: Tuple[float, float, float], v2: Tuple[float, float, float]) -> float:
        v1 = np.array(v1)
        v2 = np.array(v2)
        dot_product = np.dot(v1, v2)
        norms = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_angle = np.clip(dot_product / norms, -1.0, 1.0)
        return math.degrees(math.acos(cos_angle))

    # --- NEW triangulation function using least-squares (DROP-IN replacement) ---
    @staticmethod
    # def triangulate_least_squares(cam1: Tuple[float, float, float], dir1: Tuple[float, float, float],
    #                               cam2: Tuple[float, float, float], dir2: Tuple[float, float, float]) -> Tuple[
    #     Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float]]:
    #
    #     p1 = np.array(cam1)
    #     p2 = np.array(cam2)
    #     d1 = np.array(dir1) / np.linalg.norm(dir1)
    #     d2 = np.array(dir2) / np.linalg.norm(dir2)
    #
    #     cross = np.cross(d1, d2)
    #     denom = np.linalg.norm(cross) ** 2
    #
    #     if denom < 1e-8:
    #         mid = (p1 + p2) / 2
    #         return tuple(p1), tuple(p2), tuple(mid)
    #
    #     t = np.dot(np.cross((p2 - p1), d2), cross) / denom
    #     s = np.dot(np.cross((p2 - p1), d1), cross) / denom
    #
    #     closest_point_1 = p1 + d1 * t
    #     closest_point_2 = p2 + d2 * s
    #     mid_point = (closest_point_1 + closest_point_2) / 2
    #
    #     return tuple(closest_point_1), tuple(closest_point_2), tuple(mid_point)

    def triangulate_least_squares_with_angle(cam1: Tuple[float, float, float], dir1: Tuple[float, float, float],
                                             cam2: Tuple[float, float, float], dir2: Tuple[float, float, float]) -> \
    Tuple[
        Tuple[float, float, float], Tuple[float, float, float], Tuple[float, float, float], float, float]:

        p1 = np.array(cam1)
        p2 = np.array(cam2)
        d1 = np.array(dir1) / np.linalg.norm(dir1)
        d2 = np.array(dir2) / np.linalg.norm(dir2)

        cross = np.cross(d1, d2)
        denom = np.linalg.norm(cross) ** 2

        ray_angle_deg = Moil3dAlgorithm.angle_between_vectors(d1, d2)
        confidence = max(0.0, min(1.0, ray_angle_deg / 90.0))  # normalize to [0, 1] for visualization

        if denom < 1e-8:
            mid = (p1 + p2) / 2
            return tuple(p1), tuple(p2), tuple(mid), ray_angle_deg, confidence

        t = np.dot(np.cross((p2 - p1), d2), cross) / denom
        s = np.dot(np.cross((p2 - p1), d1), cross) / denom

        closest_point_1 = p1 + d1 * t
        closest_point_2 = p2 + d2 * s
        mid_point = (closest_point_1 + closest_point_2) / 2

        return tuple(closest_point_1), tuple(closest_point_2), tuple(mid_point), ray_angle_deg, confidence

    @staticmethod
    def get_mid_point(point_p: Tuple[float, float, float],
                      point_q: Tuple[float, float, float]) -> Tuple[float, float, float]:

        mid_point = ((point_p[0] + point_q[0]) / 2,
                     (point_p[1] + point_q[1]) / 2,
                     (point_p[2] + point_q[2]) / 2)

        return mid_point

    @staticmethod
    def nearest_2_view_points_mid_3d(alpha_l: float,
                                     beta_l: float,
                                     alpha_r: float,
                                     beta_r: float,
                                     cam_3d_coord_l: Tuple[float, float, float],
                                     cam_3d_coord_r: Tuple[float, float, float]) -> Tuple[
        Tuple[float, float, float],  # point_p
        Tuple[float, float, float],  # point_q
        Tuple[float, float, float],  # mid_point
        float,                       # ray_angle_deg
        float                        # confidence [0~1]
    ]:

        vector_l = Moil3dAlgorithm.al_ba_2_vector(alpha_l, beta_l)
        vector_r = Moil3dAlgorithm.al_ba_2_vector(alpha_r, beta_r)

        return Moil3dAlgorithm.triangulate_least_squares_with_angle(cam_3d_coord_l, vector_l,
                                                                    cam_3d_coord_r, vector_r)

    @staticmethod
    def quick_3d_measure(l_moildev, r_moildev,
                         l_cam_3d_coord: Tuple[float, float, float],
                         r_cam_3d_coord: Tuple[float, float, float],
                         l_cam_p1: Tuple[int, int],
                         l_cam_p2: Tuple[int, int],
                         r_cam_p1: Tuple[int, int],
                         r_cam_p2: Tuple[int, int]) -> float:

        left_alpha, left_beta = l_moildev.getAlphaBeta(*l_cam_p1, 1)
        left_alpha, left_beta = round(left_alpha, 1), round(left_beta, 1)

        right_alpha, right_beta = r_moildev.getAlphaBeta(*r_cam_p1, 1)
        right_alpha, right_beta = round(right_alpha, 1), round(right_beta, 1)

        point_p = Moil3dAlgorithm.nearest_2_view_points_mid_3d(left_alpha, left_beta, right_alpha, right_beta,
                                                               l_cam_3d_coord, r_cam_3d_coord)[2]

        left_alpha, left_beta = l_moildev.getAlphaBeta(*l_cam_p2, 1)
        left_alpha, left_beta = round(left_alpha, 1), round(left_beta, 1)

        right_alpha, right_beta = r_moildev.getAlphaBeta(*r_cam_p2, 1)
        right_alpha, right_beta = round(right_alpha, 1), round(right_beta, 1)

        point_q = Moil3dAlgorithm.nearest_2_view_points_mid_3d(left_alpha, left_beta, right_alpha, right_beta,
                                                               l_cam_3d_coord, r_cam_3d_coord)[2]

        distance = math.sqrt(
            pow(point_p[0] - point_q[0], 2) + pow(point_p[1] - point_q[1], 2) + pow(point_p[2] - point_q[2], 2))
        return round(distance, 2)

    @staticmethod
    def single_3d_coordinate(l_moildev, r_moildev,
                              l_cam_3d_coord: Tuple[float, float, float],
                              r_cam_3d_coord: Tuple[float, float, float],
                              l_cam_pixel_coord: Tuple[int, int],
                              r_cam_pixel_coord: Tuple[int, int], ) -> Tuple[float, float, float]:

        left_alpha, left_beta = l_moildev.getAlphaBeta(*l_cam_pixel_coord, 1)
        left_alpha, left_beta = round(left_alpha, 1), round(left_beta, 1)

        right_alpha, right_beta = r_moildev.getAlphaBeta(*r_cam_pixel_coord, 1)
        right_alpha, right_beta = round(right_alpha, 1), round(right_beta, 1)

        return Moil3dAlgorithm.nearest_2_view_points_mid_3d(left_alpha, left_beta, right_alpha, right_beta,
                                                            l_cam_3d_coord, r_cam_3d_coord)[2]



class Reprojector_3d:
    def __init__(self, cam_param: str, cam_coord: Tuple[float,float,float] = (0,0,0), cam_R: np.ndarray = None):
        """
                cam_param: path ke JSON fisheye
                cam_coord: posisi kamera di frame dunia (X,Y,Z)
                cam_R    : 3×3 matriks rotasi dari world→cam coords (None→I)
        """

        self.moil = Moildev.Moildev(cam_param)
        self.icx = self.moil.icx
        self.icy = self.moil.icy
        self._alpha2rho = lambda a: self.moil.get_rho_from_alpha(a)
        self.cam_t = np.array(cam_coord, dtype=float)
        self.cam_R = cam_R if cam_R is not None else np.eye(3)

    @staticmethod
    def _vector_from_3d(pt3d):
        x,y,z = pt3d
        r = math.sqrt(x*x + y*y + z*z)
        return x/r, y/r, z/r

    @staticmethod
    def _cartesian_beta_to_moil(beta_cart_deg: float) -> float:
        beta_moil = 90 - beta_cart_deg
        return beta_moil % 360

    def _pt3d_to_alphabeta(self, pt3d):
        # vx, vy, vz = self._vector_from_3d(pt3d)
        # alpha = math.degrees(math.acos(vz))
        # beta_cart = math.degrees(math.atan2(vy, vx))
        # beta = self._cartesian_beta_to_moil(beta_cart)
        # return alpha, beta

        # 1) transform world→camera frame: p_cam = R @ (pt3d - cam_t)
        p = np.array(pt3d, dtype=float) - self.cam_t
        x_cam, y_cam, z_cam = self.cam_R.dot(p)

        # 2) lanjutkan seperti biasa
        r = math.sqrt(x_cam * x_cam + y_cam * y_cam + z_cam * z_cam)
        vx, vy, vz = x_cam / r, y_cam / r, z_cam / r

        alpha = math.degrees(math.acos(vz))
        beta_cart = math.degrees(math.atan2(vy, vx))
        beta_moil = 90 - beta_cart
        beta = beta_moil % 360
        return alpha, beta

    def _alphabeta_to_pixel(self, alpha, beta):
        rho = self._alpha2rho(alpha)
        beta_cart = 90 - beta
        u = self.icx + rho * math.cos(math.radians(beta_cart))
        v = self.icy - rho * math.sin(math.radians(beta_cart))
        return u,v

    def _reproject_point(self, pt3d):
        alpha, beta = self._pt3d_to_alphabeta(pt3d)
        return self._alphabeta_to_pixel(alpha, beta)

#
# class Moil3dAlgorithm:
#
#     @staticmethod
#     def _beta_moil2cartesian(beta_moil: float) -> float:
#         beta_cartesian = beta_moil % 360
#         beta_cartesian = 90 - beta_cartesian
#
#         if beta_cartesian < 0:
#             return beta_cartesian + 360
#         else:
#             return beta_cartesian
#
#     @staticmethod
#     def al_ba_2_vector_x(alpha: float, beta: float) -> float:
#         beta = Moil3dAlgorithm._beta_moil2cartesian(beta)
#         return math.sin(math.radians(alpha)) * math.cos(math.radians(beta))
#
#     @staticmethod
#     def al_ba_2_vector_y(alpha: float, beta: float) -> float:
#         beta = Moil3dAlgorithm._beta_moil2cartesian(beta)
#         return math.sin(math.radians(alpha)) * math.sin(math.radians(beta))
#
#     @staticmethod
#     def al_2_vector_z(alpha: float) -> float:
#         return math.cos(math.radians(alpha))
#
#     @staticmethod
#     def al_ba_2_vector(alpha: float, beta: float) -> Tuple[float, float, float]:
#         vector_x = Moil3dAlgorithm.al_ba_2_vector_x(alpha, beta)
#         vector_y = Moil3dAlgorithm.al_ba_2_vector_y(alpha, beta)
#         vector_z = Moil3dAlgorithm.al_2_vector_z(alpha)
#
#         return vector_x, vector_y, vector_z
#
#     @staticmethod
#     def get_unknown_3d_coord_and_unknown_symbol(vector: Tuple[float, float, float],
#                                                 cam_3d_coord: Tuple[float, float, float],
#                                                 symbol_name: str) -> Tuple[Tuple[float, float, float], Symbol]:
#
#         vector_x = vector[0]
#         vector_y = vector[1]
#         vector_z = vector[2]
#
#         cam_3d_coord_x = cam_3d_coord[0]
#         cam_3d_coord_y = cam_3d_coord[1]
#         cam_3d_coord_z = cam_3d_coord[2]
#
#         unknown_symbol = Symbol(symbol_name)
#         unknown_coord = (vector_x * unknown_symbol + cam_3d_coord_x,
#                          vector_y * unknown_symbol + cam_3d_coord_y,
#                          vector_z * unknown_symbol + cam_3d_coord_z)
#
#         return unknown_coord, unknown_symbol
#
#     @staticmethod
#     def solve_unknown_symbol_m_n(left_vector: Tuple[float, float, float],
#                                  right_vector: Tuple[float, float, float],
#                                  unknown_coord_p: Tuple[float, float, float],
#                                  unknown_coord_q: Tuple[float, float, float],
#                                  unknown_symbol_m: Symbol,
#                                  unknown_symbol_n: Symbol) -> Tuple[Symbol, Symbol]:
#
#         vector_pq_x = unknown_coord_q[0] - unknown_coord_p[0]
#         vector_pq_y = unknown_coord_q[1] - unknown_coord_p[1]
#         vector_pq_z = unknown_coord_q[2] - unknown_coord_p[2]
#
#         left_vector_x = left_vector[0]
#         left_vector_y = left_vector[1]
#         left_vector_z = left_vector[2]
#
#         right_vector_x = right_vector[0]
#         right_vector_y = right_vector[1]
#         right_vector_z = right_vector[2]
#
#         eq1 = Eq((vector_pq_x * left_vector_x +
#                   vector_pq_y * left_vector_y +
#                   vector_pq_z * left_vector_z), 0)
#
#         eq2 = Eq((vector_pq_x * right_vector_x +
#                   vector_pq_y * right_vector_y +
#                   vector_pq_z * right_vector_z), 0)
#
#         ans_s_t = solve((eq1, eq2), (unknown_symbol_m, unknown_symbol_n))
#
#         unknown_symbol_m = ans_s_t[unknown_symbol_m]
#         unknown_symbol_n = ans_s_t[unknown_symbol_n]
#
#         return unknown_symbol_m, unknown_symbol_n
#
#     @staticmethod
#     def calculate_closest_2_3d_coord_of_2_view_line(left_vector: Tuple[float, float, float],
#                                                     right_vector: Tuple[float, float, float],
#                                                     cam_3d_coord_l: Tuple[float, float, float],
#                                                     cam_3d_coord_r: Tuple[float, float, float],
#                                                     n: Symbol,
#                                                     m: Symbol) \
#             -> Tuple[Tuple[float, float, float],
#                      Tuple[float, float, float]]:
#
#         point_p = (left_vector[0] * n + cam_3d_coord_l[0],
#                    left_vector[1] * n + cam_3d_coord_l[1],
#                    left_vector[2] * n + cam_3d_coord_l[2])
#
#         point_q = (right_vector[0] * m + cam_3d_coord_r[0],
#                    right_vector[1] * m + cam_3d_coord_r[1],
#                    right_vector[2] * m + cam_3d_coord_r[2])
#
#         return point_p, point_q
#
#     @staticmethod
#     def get_mid_point(point_p: Tuple[float, float, float],
#                       point_q: Tuple[float, float, float]) -> Tuple[float, float, float]:
#
#         mid_point = ((point_p[0] + point_q[0]) / 2,
#                      (point_p[1] + point_q[1]) / 2,
#                      (point_p[2] + point_q[2]) / 2)
#
#         return mid_point
#
#     @staticmethod
#
#     def nearest_2_view_points_mid_3d(alpha_l: float,
#                               beta_l: float,
#                               alpha_r: float,
#                               beta_r: float,
#                               cam_3d_coord_l: Tuple[float, float, float],
#                               cam_3d_coord_r: Tuple[float, float, float]) -> Tuple[
#         Tuple[float, float, float],  # point_p
#         Tuple[float, float, float],  # point_q
#         Tuple[float, float, float]  # mid_point
#     ]:
#
#         vector_l = Moil3dAlgorithm.al_ba_2_vector(alpha_l, beta_l)
#
#         vector_r = Moil3dAlgorithm.al_ba_2_vector(alpha_r, beta_r)
#
#         point_p, m = Moil3dAlgorithm.get_unknown_3d_coord_and_unknown_symbol(vector_l,
#                                                                              cam_3d_coord_l, 'm')
#
#         point_q, n = Moil3dAlgorithm.get_unknown_3d_coord_and_unknown_symbol(vector_r,
#                                                                              cam_3d_coord_r, 'n')
#
#         m, n = Moil3dAlgorithm.solve_unknown_symbol_m_n(vector_l,
#                                                         vector_r,
#                                                         point_p, point_q, m, n)
#
#         point_p, point_q = Moil3dAlgorithm.calculate_closest_2_3d_coord_of_2_view_line(vector_l,
#                                                                                        vector_r,
#                                                                                        cam_3d_coord_l,
#                                                                                        cam_3d_coord_r,
#                                                                                        m, n)
#
#         mid_point = Moil3dAlgorithm.get_mid_point(point_p, point_q)
#
#         return point_p, point_q, mid_point
#
#     @staticmethod
#     def quick_3d_measure(l_moildev, r_moildev,
#                          l_cam_3d_coord: Tuple[float, float, float],
#                          r_cam_3d_coord: Tuple[float, float, float],
#                          l_cam_p1: Tuple[int, int],
#                          l_cam_p2: Tuple[int, int],
#                          r_cam_p1: Tuple[int, int],
#                          r_cam_p2: Tuple[int, int]) -> float:
#
#         """
#         :param l_moildev: Moildev object (cam_l)
#         :param r_moildev: Moildev object (cam_r)
#         :param l_cam_3d_coord: Tuple[x: float, y: float, z: float] 3d coord (cam_l)
#         :param r_cam_3d_coord: Tuple[x: float, y: float, z: float] 3d coord (cam_r)
#         :param l_cam_p1: Tuple[x: int, y: int] pixel coord "p1" (cam_l)
#         :param l_cam_p2: Tuple[x: int, y: int] pixel coord "p2" (cam_r)
#         :param r_cam_p1: Tuple[x: int, y: int] pixel coord "p1" (cam_l)
#         :param r_cam_p2: Tuple[x: int, y: int] pixel coord "p2" (cam_r)
#         :return: float
#         """
#
#         left_alpha, left_beta = l_moildev.getAlphaBeta(*l_cam_p1, 1)
#         left_alpha, left_beta = round(left_alpha, 1), round(left_beta, 1)
#
#         right_alpha, right_beta = r_moildev.getAlphaBeta(*r_cam_p1, 1)
#         right_alpha, right_beta = round(right_alpha, 1), round(right_beta, 1)
#
#         point_p = Moil3dAlgorithm.nearest_mid_3d_coord(left_alpha, left_beta, right_alpha, right_beta,
#                                                        l_cam_3d_coord, r_cam_3d_coord)
#
#         left_alpha, left_beta = l_moildev.getAlphaBeta(*l_cam_p2, 1)
#         left_alpha, left_beta = round(left_alpha, 1), round(left_beta, 1)
#
#         right_alpha, right_beta = r_moildev.getAlphaBeta(*r_cam_p2, 1)
#         right_alpha, right_beta = round(right_alpha, 1), round(right_beta, 1)
#
#         point_q = Moil3dAlgorithm.nearest_mid_3d_coord(left_alpha, left_beta, right_alpha, right_beta,
#                                                        l_cam_3d_coord, r_cam_3d_coord)
#         distance = math.sqrt(
#             pow(point_p[0] - point_q[0], 2) + pow(point_p[1] - point_q[1], 2) + pow(point_p[2] - point_q[2], 2))
#         return round(distance, 2)
#
#
#
#     @staticmethod
#     def single_3d_coordinate(l_moildev, r_moildev,
#                          l_cam_3d_coord: Tuple[float, float, float],
#                          r_cam_3d_coord: Tuple[float, float, float],
#                          l_cam_pixel_coord: Tuple[int, int],
#                          r_cam_pixel_coord: Tuple[int, int],) -> Tuple[float, float, float]:
#
#         """
#         :param l_moildev: Moildev object (cam_l)
#         :param r_moildev: Moildev object (cam_r)
#         :param l_cam_3d_coord: Tuple[x: float, y: float, z: float] 3d coord (cam_l)
#         :param r_cam_3d_coord: Tuple[x: float, y: float, z: float] 3d coord (cam_r)
#         :param l_cam_pixel_coord: Tuple[x: int, y: int] pixel coord (cam_l)
#         :param r_cam_pixel_coord: Tuple[x: int, y: int] pixel coord (cam_r)
#         :return: Tupple
#
#         This method calculates the single coordinate of the target on the image.
#         Only the single corresponding pixel coordinates on the left and right images need to be selected.
#         """
#
#         left_alpha, left_beta = l_moildev.getAlphaBeta(*l_cam_p1, 1)
#         left_alpha, left_beta = round(left_alpha, 1), round(left_beta, 1)
#
#         right_alpha, right_beta = r_moildev.getAlphaBeta(*r_cam_p1, 1)
#         right_alpha, right_beta = round(right_alpha, 1), round(right_beta, 1)
#
#         return Moil3dAlgorithm.nearest_mid_3d_coord(left_alpha, left_beta, right_alpha, right_beta,
#                                                        l_cam_3d_coord, r_cam_3d_coord)
