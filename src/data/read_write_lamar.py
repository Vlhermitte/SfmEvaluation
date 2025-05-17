import open3d as o3d
import numpy as np
from scipy.spatial.transform import Rotation as R
from pathlib import Path
from typing import Iterator, List, Union, Optional
from tqdm import tqdm
import shutil
from concurrent.futures import ThreadPoolExecutor, as_completed

from data.read_write_model import write_model, Camera as ColmapCamera, Image as ColmapImage, Point3D as ColmapPoint3D


def copy_image(image_path, output_dir):
    shutil.copy2(image_path, output_dir / image_path.name)


class LamarSessionProcessor:
    """
    Class to read session trajectories from LaMAR dataset and export them to COLMAP format.
    Attributes
    ----------
    datasets_path : Path
        Path to the LaMAR dataset.
    """
    def __init__(self, datasets_path: Path):
        self.datasets_path = Path(datasets_path)

    def get_trajectory(self, session_id: str):
        # Read the file
        session = self.datasets_path / 'sessions/map/raw_data/' / session_id
        assert session.exists(), f"Error: {session} does not exist"

        # Read the file trajectories.txt
        trajectories = self.datasets_path / 'sessions/map/trajectories.txt'

        # Each line is format (timestamp, device_id, qw, qx, qy, qz, tx, ty, tz, *covar)
        # Select only the line with device_id = session (e.g. ios_2022-01-12_14.59.02/came_phone_*)
        poses = {}
        with open(trajectories, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
            for line in lines:
                data = line.split()
                # split data[1] to get the session_id
                device_id = data[1].split('/')[0]
                if device_id == session_id:
                    timestamps = int(data[0].split(',')[0])
                    pose_data = [float(x.split(',')[0]) for x in data[2:]]
                    q_cam2world, t_cam2world, covar = pose_data[:4], pose_data[4:7], pose_data[7:]

                    # Quaternion and translation are in camera to world coordinate frame. Convert to world to camera
                    rot = R.from_quat([q_cam2world[1], q_cam2world[2], q_cam2world[3],
                                       q_cam2world[0]])  # Convert [w, x, y, z] -> [x, y, z, w]
                    inv_rot = rot.inv()
                    q = inv_rot.as_quat()
                    q_world2cam = [q[3], q[0], q[1], q[2]]
                    t_world2cam = -inv_rot.apply(t_cam2world)
                    poses[timestamps] = (q_world2cam, t_world2cam)

        return poses

    def get_sensors(self, session_id):
        # Read the file sensors.txt
        sensors = self.datasets_path / 'sessions/map/sensors.txt'

        # Each line is format (sensor_id, name, sensor_type, [sensor_params]+)
        # Select only the line with device_id = session (e.g. ios_2022-01-12_14.59.02/came_phone_*)
        cameras_sensors = []
        with open(sensors, 'r') as f:
            lines = f.readlines()
            lines = lines[1:]
            for line in lines:
                data = line.split(', ')
                # split data[1] to get the session_id
                sensor_id = data[0].split('/')[0]
                if sensor_id == session_id and data[2] == 'camera':
                    name = int(data[1].split()[-1])
                    sensor_type = data[3]
                    width, height, params = int(data[4]), int(data[5]), [float(x) for x in data[6:]]
                    cameras_sensors.append(
                        ColmapCamera(
                            id=name,
                            model=sensor_type,
                            width=width,
                            height=height,
                            params=params
                        )
                    )

        return cameras_sensors

    def get_images_path(self, session_ids: Union[str, List[str]]):
        if isinstance(session_ids, str):
            session_ids = [session_ids]

        images_paths = []
        for session_id in session_ids:
            session_path = self.datasets_path / 'sessions/map/raw_data' / session_id
            assert session_path.exists(), f"Error: {session_path} does not exist"

            images_dir = session_path / 'images'
            images_paths.append((session_id, images_dir))

        return images_paths

    def export_to_colmap(self, session_ids: Union[str, List[str]], saving_path: Optional[Path] = None):
        """
        Export camera trajectories for one or multiple sessions into COLMAP format files.

        Parameters
        ----------
        session_ids : Union[str, List[str]]
            A single session ID or a list of session IDs.
        saving_path : Optional[Path]
            Path to save the COLMAP files. If None, the default path is used.
        """
        # Convert single session ID to a list
        if isinstance(session_ids, str):
            session_ids = [session_ids]

        # Process each session.
        with tqdm(session_ids, desc='Exporting sessions') as pbar:
            for session_id in pbar:
                pbar.set_description(f'Exporting session: {session_id}')
                session_path = self.datasets_path / 'sessions/map/raw_data' / session_id
                assert session_path.exists(), f"Error: {session_path} does not exist"

                colmap_cameras = {}
                colmap_images = {}
                image_index = 0

                # Get the trajectory and sensors for this session.
                poses = self.get_trajectory(session_id)
                sensors = self.get_sensors(session_id)
                # Merge sensors into a single dictionary.
                for camera in sensors:
                    colmap_cameras[camera.id] = camera

                # Create COLMAP image entries for each pose.
                # We prepend the session ID to the image name for uniqueness.
                for timestamp, (q, t) in poses.items():
                    colmap_images[image_index] = ColmapImage(
                        id=image_index,
                        qvec=q,
                        tvec=t,
                        camera_id=int(timestamp),
                        name=f'{timestamp}.jpg',
                        xys=[],
                        point3D_ids=[]
                    )
                    image_index += 1

                # COLMAP points3D.txt remains empty.
                colmap_points = {}

                # Use a dedicated output folder for the combined export.
                if saving_path is None:
                    output_path = session_path / 'sparse' / '0'
                    output_path.mkdir(exist_ok=True, parents=True)
                else:
                    output_path = Path(saving_path)
                write_model(colmap_cameras, colmap_images, colmap_points, output_path, ext='.txt')

    def export_all_in_directory(self, session_ids: Union[str, List[str]]):
        if isinstance(session_ids, str):
            session_ids = [session_ids]

        for session_id in session_ids:
            session_path = self.datasets_path / 'sessions/map/raw_data' / session_id
            assert session_path.exists(), f"Error: {session_path} does not exist"

            output_path = self.datasets_path / 'sessions/map' / 'colmap_all' / 'images'
            output_path.mkdir(exist_ok=True, parents=True)

            images_dir = self.datasets_path / 'sessions/map/raw_data' / session_id / 'images'
            image_paths = list(images_dir.glob('*.jpg'))

            with ThreadPoolExecutor(max_workers=8) as executor:
                futures = {executor.submit(copy_image, image, output_path): image for image in image_paths}
                for future in tqdm(as_completed(futures), total=len(futures),
                                   desc=f'Exporting images from {session_id}'):
                    try:
                        future.result()
                    except Exception as e:
                        print(f"Error copying {futures[future]}: {e}")


def visualize_trajectory(trajectory):
    vis = o3d.visualization.Visualizer()
    vis.create_window()

    # Create coordinate frames and line sets
    frames = []
    points = []
    lines = []

    for i, (_, quaternion, position) in enumerate(trajectory):
        # Create coordinate frame
        frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
        rotation = o3d.geometry.get_rotation_matrix_from_quaternion(quaternion)
        frame.rotate(rotation, center=(0, 0, 0))
        frame.translate(position)
        frames.append(frame)

        points.append(position)
        if i > 0:
            lines.append([i - 1, i])

    # Combine frames
    for frame in frames:
        vis.add_geometry(frame)

    # Create trajectory line
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(points),
        lines=o3d.utility.Vector2iVector(lines)
    )
    line_set.colors = o3d.utility.Vector3dVector([[1, 0, 0] for _ in lines])  # red lines
    vis.add_geometry(line_set)

    vis.run()
    vis.destroy_window()


if __name__ == '__main__':
    script_dir = Path(__file__).resolve().parent
    datasets_path = script_dir / '../../data/datasets/LaMAR/HGE'
    datasets_path = datasets_path.resolve()
    if not datasets_path.exists():
        print("Error: datasets_path does not exist")
        exit(1)

    # retrieve all session ids that start with ios
    # session_ids = [session.name for session in datasets_path.glob('sessions/map/raw_data/ios*')]
    session_ids = [session.name for session in datasets_path.glob('sessions/map/raw_data/ios*')]
    lamar = LamarSessionProcessor(datasets_path)
    lamar.export_to_colmap(session_ids)
    # lamar.export_all_in_directory(session_ids)
