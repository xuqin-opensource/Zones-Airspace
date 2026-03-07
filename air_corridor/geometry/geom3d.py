from air_corridor.tools._descriptor import Direction, Position, PositiveNumber
from air_corridor.tools._geometric import Geometric3D
from air_corridor.tools.util import *


# @abstractmethod
# def project_point(self, point):
# pass
class Point3D(Geometric3D):
    anchor_point = Position(3)
    orientation_vec = Direction(3)

    def __init__(self, anchor_point=np.array([0, 0, 0]), orientation_vec=None, orientation_rad=None):
        self.anchor_point = anchor_point
        # can be considered as projected z

        assert orientation_rad is not None or orientation_vec is not None
        if orientation_vec is None:
            # theta, phi = orientation_rad
            # self.orientation_rad = orientation_rad
            self.orientation_vec = polar_to_unit_normal(orientation_rad) # 将球坐标转换为方向向量
        else:
            self.orientation_vec = orientation_vec

        # projected x from base x
        # 将全局坐标系中Z轴旋转为orientation_vec，得到局部坐标系中x轴方向
        self.x = rotate(vec=X_UNIT, fromVec=Z_UNIT, toVec=self.orientation_vec)

        # projected y from base x
        self.y = np.cross(self.orientation_vec, self.x) # 局部坐标系中y轴方向为x方向叉乘
        # 基坐标系（Base Frame）：参考坐标系，Z轴方向为基准
        # 局部（走廊）坐标系（Remote Frame）：实际物体（如走廊）的本地坐标系
        self.rotation_matrix_to_remote = vec2vec_rotation(Z_UNIT, self.orientation_vec) # 从Z轴到orientation_vec的旋转矩阵
        self.rotation_matrix_to_base = vec2vec_rotation(self.orientation_vec, Z_UNIT) # 从orientation_vec到Z轴的旋转矩阵

    def rotate_to_base(self, vec):
        # 将局部（走廊）坐标系中的向量转换到基坐标系
        return np.dot(self.rotation_matrix_to_base, vec)

    def rotate_to_remote(self, vec):
        # 将基坐标系中的向量转换到局部（走廊）坐标系
        return np.dot(self.rotation_matrix_to_remote, vec)

    def project_to_base(self, point):
        vec = self.point_relative_center_position(point)    # 计算点相对于锚点的位置向量
        return self.rotate_to_base(vec) # 将向量旋转到基坐标系

    def point_relative_center_position(self, point):
        return point - self.anchor_point

    # @lru_cache(maxsize=2)
    def convert_2_polar(self, point, reduce_space):
        if reduce_space:
            point = self.project_to_base(point) # 点→相对位置→基坐标系→球坐标
        else:
            point = self.point_relative_center_position(point)  # 点→相对位置→球坐标
        r, theta, phi = cartesian_to_polar_or_spherical(point)
        return r, theta, phi

    def convert_vec_2_polar(self, point, velocity, reduce_space):
        # 用于将速度向量转换为球坐标系下的变化率
        r1, theta1, phi1 = self.convert_2_polar(point, reduce_space)
        r2, theta2, phi2 = self.convert_2_polar(point + velocity, reduce_space)
        return r2 - r1, theta2 - theta1, phi2 - phi1

    def cartesian_to_polar(self, point):
        """ 1) convert to relative position and 2) then convert to polar coordinate """
        relative_position = self.point_relative_center_position(point)
        return cartesian_to_polar_or_spherical(relative_position)

    def is_inside(self, point):
        return np.allclose(point, self.anchor_point)

    def report(self, base=None):
        # only fit reduce_space scenario
        if self != base:
            ori_based_on_former = base.rotate_to_base(self.orientation_vec)
            _, theta, phi = cartesian_to_polar_or_spherical(ori_based_on_former)
            status = list(base.anchor_point - self.anchor_point) +\
                      list(base.rotate_to_base(self.orientation_vec)) +\
                      [theta, phi]
        else:
            status = [0, 0, 0] +list(Z_UNIT) +[0, 0]
        return status


class Sphere(Point3D):
    radius = PositiveNumber()

    def __init__(self, anchor_point, orientation_vec, radius):
        super().__init__(anchor_point, orientation_vec)
        self.radius = radius

    def __repr__(self):
        return f"Sphere(center={self.anchor_point.tolist()}, " \
               f"radius={self.radius})"

    def distance_object_to_point(self, point):
        return super().point_relative_center_position(point) - self.radius

    def is_inside(self, point):
        return True if self.distance_object_to_point(point) < TRIVIAL_TOLERANCE else False


class Cylinder(Point3D):
    radius = PositiveNumber()
    length = PositiveNumber()

    def __init__(self, anchor_point, orientation_vec, orientation_rad, radius, length=1):
        super().__init__(anchor_point,
                         orientation_vec=orientation_vec,
                         orientation_rad=orientation_rad)
        self.radius = radius
        self.length = length

        self.endCirclePlane = (
            Circle(
                anchor_point=self.anchor_point + self.orientation_vec * self.length / 2,
                orientation_vec=self.orientation_vec,
                radius=radius
            )
        )

    def __repr__(self):
        return f"Cylinder(anchor_point={self.anchor_point.tolist()}, " \
               f"orientation_vec={self.orientation_vec.tolist()}, " \
               f"radius={self.radius}," \
               f"length={self.length})"

    def distance_object_to_point(self, point):
        # y: perpendicular to the line; x: parallel/projected to the line
        distance_y = distance_perpendicular_line_point(self.anchor_point, self.orientation_vec, point) - self.radius
        distance_x = np.abs(
            distance_signed_parallel_line_point(self.anchor_point, self.orientation_vec, point)) - self.length / 2
        return max(distance_x, distance_y)

    def is_inside(self, point):
        return (True,None) if self.distance_object_to_point(point) <= TRIVIAL_TOLERANCE else (False,'breached')

    def line_cross_des_plane_n_how_much(self, inside_point, outside_point):
        if self.is_inside(self.point_relative_center_position(outside_point)):
            raise Exception("outside point is not outside")
        return is_line_line_intersect(self.point_relative_center_position(inside_point),
                                      self.point_relative_center_position(outside_point),
                                      self.up_left,
                                      self.up_right)


class Circle(Point3D):
    def __init__(self, anchor_point, orientation_vec, radius):
        super().__init__(anchor_point, orientation_vec)
        self.radius = radius
        if any(np.isnan(self.x)) or any(np.isnan(self.y)):
            print(1)
            super().__init__(anchor_point, orientation_vec)

    def cross_circle_plane(self, line_start, line_end):
        return is_line_circle_intersect(line_start=line_start,
                                        line_end=line_end,
                                        anchor=self.anchor_point,
                                        direction=self.orientation_vec,
                                        radius=self.radius)

    def distance_object_to_point(self, point):
        pass

    def report_state(self):
        pass


class newTorus(Point3D):
    major_radius = PositiveNumber() # 主半径验证（必须为正数）
    minor_radius = PositiveNumber()

    def __init__(self,
                 anchor_point,
                 orientation_vec,   # 环面法向量（指定环面朝向）
                 orientation_rad,   # 方向角度（备用表示）
                 major_radius,      # 主半径（从中心到管中心的距离）
                 minor_radius,      # 次半径（管本身的半径）
                 begin_rad,         # 起始弧度（环的起点角度）
                 end_rad):          # 结束弧度（环的终点角度）
        super().__init__(anchor_point,
                         orientation_vec=orientation_vec,
                         orientation_rad=orientation_rad)
        self.begin_rad = begin_rad
        self.end_rad = end_rad

        self.beginCirclePlane = (
            Circle(
                # 从环面中心 anchor_point 出发，沿局部xy平面移动 major_radius 距离
                # 方向由角度 begin_rad 决定：self.x*cos(begin_rad): x方向分量，self.y*sin(begin_rad): y方向分量
                anchor_point=self.anchor_point + major_radius * (
                        self.x * np.cos(begin_rad) + self.y * np.sin(begin_rad)),
                orientation_vec=(-self.x * np.sin(begin_rad) + self.y * np.cos(begin_rad)), # 初始圆平面法向量
                radius=minor_radius
            )
        )
        self.endCirclePlane = (
            Circle(
                anchor_point=self.anchor_point + major_radius * (
                        self.x * np.cos(end_rad) + self.y * np.sin(end_rad)),
                orientation_vec=(-self.x * np.sin(end_rad) + self.y * np.cos(end_rad)),
                radius=minor_radius
            )
        )

        self.major_radius = major_radius
        self.minor_radius = minor_radius

        # attatch begin radian to 0 rad
        # self.rotate_xy_begin_to_x = o3d.geometry.get_rotation_matrix_from_axis_angle(-self.begin_rad * Z_UNIT)
        # self.rotate_xy_x_to_begin = o3d.geometry.get_rotation_matrix_from_axis_angle(+self.begin_rad * Z_UNIT)

        # 每个环形坡道（Torus）有自己的起始角度 begin_rad 和结束角度 end_rad
        # 为了简化连接其他几何体，需要将环形坡道标准化到一个固定角度
        # attatch end radian to pi/2 rad
        # 将轴角法转换为旋转矩阵
        #方向(旋转轴)：Z_UNIT = [0, 0, 1]（z轴），旋转角度：(+np.pi/2 - self.end_rad) 弧度
        self.rotate_xy_begin_to_x = o3d.geometry.get_rotation_matrix_from_axis_angle(
            (+np.pi / 2 - self.end_rad) * Z_UNIT)   # 这个矩阵将实际终止角度旋转到标准角度π/2
        self.rotate_xy_x_to_begin = o3d.geometry.get_rotation_matrix_from_axis_angle(
            (-np.pi / 2 + self.end_rad) * Z_UNIT)   # 这个矩阵将标准角度π/2旋转回实际终止角度

        # 复合变换到世界坐标系，矩阵乘法顺序：
        # rotate_torus_to_base = rotate_xy_begin_to_x · rotation_matrix_to_base
        # 先：self.rotation_matrix_to_base将点从局部坐标系转换到世界坐标系
        # 后：self.rotate_xy_begin_to_x将终止角度标准化到π/2
        self.rotate_torus_to_base = np.dot(self.rotate_xy_begin_to_x, self.rotation_matrix_to_base)
        # 复合变换从世界坐标系（与上文相反）
        self.rotate_torus_to_remote = np.dot(self.rotation_matrix_to_remote, self.rotate_xy_x_to_begin)

    def rotate_to_base(self, vec):
        return np.dot(self.rotate_torus_to_base, vec)

    def rotate_to_remote(self, vec):
        return np.dot(self.rotate_torus_to_remote, vec)

    def project_to_base(self, point):
        vec = self.point_relative_center_position(point)
        return self.rotate_to_base(vec)

    def __repr__(self):
        return f"Torus(center={self.anchor_point.tolist()}, " \
               f"orientation_vec={self.orientation_vec.tolist()}, " \
               f"major_radius={self.major_radius}, " \
               f"minor_radius={self.minor_radius}, " \
               f"begin_degree={self.begin_rad}, " \
               f"end_degree={self.end_rad})"

    def report_state(self):
        return self.anchor_point.tolist() + self.orientation_vec.tolist() + [self.major_radius, self.minor_radius,
                                                                             self.begin_rad, self.end_rad]

    def determine_positive_direction(self, point):
        '''
        out put the positive direction based on current position
        :param point:
        :return:
        '''
        vec_to_point = self.point_relative_center_position(point)   # point - self.anchor_point
        orientation_vec = np.cross(self.orientation_vec, vec_to_point)  # self.orientation_vec是环面法向量，垂直于环面锚点所在平面

        return orientation_vec / np.linalg.norm(orientation_vec)

    def distance_object_to_point(self, point, consider_angle=False):
        '''
        1. Project the Point onto the Plane of the Circle
        2. Find the Closest Point on the Full Circle
        3. Check if the Closest Point is within the Quarter Circle Segment
        '''
        # Project the point onto the plane of the circle

        vec_to_point = self.point_relative_center_position(point)
        projection_on_plane = proj_to_plane(vec_to_point, self.orientation_vec)
        unit_projection = projection_on_plane / np.linalg.norm(projection_on_plane)

        # Closest point on the full circle
        closest_on_circle = self.anchor_point + self.major_radius * unit_projection
        signed_distance = np.linalg.norm(point - closest_on_circle) - self.minor_radius

        if not consider_angle:
            return signed_distance

        angle = np.arctan2(np.dot(closest_on_circle - self.anchor_point, self.y),
                           np.dot(closest_on_circle - self.anchor_point, self.x))
        # degree_inside = self.is_degree_in(angle)
        degree_inside = self.is_angle_in_arc(angle, self.begin_rad, self.end_rad, direction='shortest')

        return signed_distance, degree_inside
    
   

    def is_angle_in_arc(self, angle, begin_rad, end_rad, direction='ccw', inclusive=True):
        """
        判断 angle 是否在由 begin_rad -> end_rad 指定的圆弧内。
        参数:
        angle, begin_rad, end_rad: 弧度（可为任意实数，内部会归一化）
        direction: 'ccw'（逆时针增加角度）, 'cw'（顺时针）, 'shortest'（取较短弧）
        inclusive: 边界是否包含（默认包含）
        返回:
        bool
        说明:
        若 begin_rad == end_rad，函数返回 True（认为为整圈）。
        """
        def _to_0_2pi(angle):
            return angle % (2 * np.pi)
        
        a = _to_0_2pi(angle)
        b = _to_0_2pi(begin_rad)
        e = _to_0_2pi(end_rad)

        # treat begin==end as full circle
        if np.isclose((e - b) % (2*np.pi), 0.0):
            return True

        if direction == 'shortest':
            ccw_len = (e - b) % (2*np.pi)
            direction = 'ccw' if ccw_len <= np.pi else 'cw'

        if direction == 'ccw':
            if b <= e:
                if inclusive:
                    return (b <= a <= e)
                else:
                    return (b < a < e)
            else:  # wrap
                if inclusive:
                    return (a >= b) or (a <= e)
                else:
                    return (a > b) or (a < e)

        if direction == 'cw':
            # cw from begin to end is equivalent to ccw from end to begin
            if e <= b:
                if inclusive:
                    return (e <= a <= b)
                else:
                    return (e < a < b)
            else:  # wrap
                if inclusive:
                    return (a >= e) or (a <= b)
                else:
                    return (a > e) or (a < b)

        raise ValueError("direction must be 'ccw', 'cw' or 'shortest'")

    # def is_degree_in(self, angle):
    #     '''
    #     always incurs a lot of bugs
    #     range for self.begin_rad is [-np.pi,np.pi]
    #     range for self.end_rad for [-np.pi, 2+np.pi]

    #     :param angle:
    #     :return:
    #     '''

    #     # assert self.end_rad > self.begin_rad

    #     while angle <self.begin_rad:
    #         angle+=np.pi * 2

    #     if self.begin_rad <= angle and angle <= self.end_rad:
    #         return True
    #     else:
    #         return False

    def is_inside(self, point):
        signed_distance, degree_inside = self.distance_object_to_point(point, consider_angle=True)
        status=[]
        # if degree_inside and signed_distance <= TRIVIAL_TOLERANCE:
        #     return True
        # else:
        #     return False

        if not degree_inside:
            status.append('rad')
        if not signed_distance <= TRIVIAL_TOLERANCE:
            status.append('wall')
        if status:
            return False,f"breached_{'_'.join(status)}"
        else:
            return True, None

