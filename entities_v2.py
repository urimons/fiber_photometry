import math
import matplotlib.pyplot as plt
from datetime import date
from typing import Optional, List, Tuple, Dict
import datetime
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np


class GCaMPData:
    def __init__(self, GCaMP_raw, Isosbestic_raw, GCaMP_dF_F, time_seconds, z_score, sample_rate):
        self.GCaMP_raw = GCaMP_raw
        self.Isosbestic_raw = Isosbestic_raw
        self.GCaMP_dF_F = GCaMP_dF_F
        self.time = time_seconds
        self.sample_rate = sample_rate
        self.z_score = z_score


    def downsample(self, new_rate):
        factor = self.sample_rate / new_rate
        if factor != int(factor):
            raise ValueError("New sample rate must be a divisor of the original sample rate.")
        
        factor = int(factor)
        self.GCaMP_raw = self.GCaMP_raw[::factor]
        self.Isosbestic_raw = self.Isosbestic_raw[::factor]
        self.GCaMP_dF_F = self.GCaMP_dF_F[::factor]
        self.time = self.time[::factor]
        self.sample_rate = new_rate

    def __repr__(self):
        return (f"GCaMPData(GCaMP_raw=<{len(self.GCaMP_raw)} values>, "
                f"Isosbestic_raw=<{len(self.Isosbestic_raw)} values>, "
                f"GCaMP_dF_F=<{len(self.GCaMP_dF_F)} values>, time=<{len(self.time)} values>, "
                f"z_score=<{len(self.z_score)} values>, sample_rate={self.sample_rate})")
    
    def to_dict(self):
        return {
            "GCaMP_raw": self.GCaMP_raw,
            "Isosbestic_raw": self.Isosbestic_raw,
            "GCaMP_dF_F": self.GCaMP_dF_F,
            "time": self.time,
            "z_score": self.z_score,
            "sample_rate": self.sample_rate
        }
    

class LimbData:
    def __init__(self, x, y, probability, name):
        self.x = x
        self.y = y
        self.probability = probability
        self.name = name

    def __repr__(self):
        return f"LimbData(name={self.name} x={self.x}, y={self.y}, probability={self.probability})"

    def to_dict(self):
        return {f'{self.name}_x': self.x, f'{self.name}_y': self.y, f'{self.name}_probability': self.probability}




class Frame:
    def __init__(self, limbs, frame_number):
        """
        Initialize the Frame object.

        :param limbs: List of LimbData objects
        :param pixel_size_mm: Pixel size in millimeters
        """
        self.limbs = limbs  # List of LimbData objects
        self.frame_number = frame_number  # Frame number
        self.pixel_size_mm = self.calculate_pixel_size_mm()  # Pixel size in millimeters

    def __repr__(self):
        return f"Frame(frame_number={self.frame_number}, pixel_size_mm={self.pixel_size_mm}, limbs={self.limbs})"

    def calculate_pixel_size_mm(self):
        """
        Calculate the pixel size in millimeters based on the distance between 'R_corner' and 'L_corner' limbs.
        """
        r_corner = next((limb for limb in self.limbs if limb.name == 'R_corner'), None)
        l_corner = next((limb for limb in self.limbs if limb.name == 'L_corner'), None)
        
        if r_corner and l_corner:
            distance_px = math.sqrt((r_corner.x - l_corner.x) ** 2 + (r_corner.y - l_corner.y) ** 2)
            # 20 cm is 200 mm, so the calculation is: 200 mm / distance in pixels
            pixel_size_mm = 200 / distance_px
            return pixel_size_mm
        else:
            return None  # Or some default value if the corners are not present


    def to_dict(self):
        """
        Convert Frame object to a dictionary representation.
        """
        limbs_dicts = [limb.to_dict() for limb in self.limbs]
        frame_dict = {
            'frame_number': self.frame_number,
            'pixel_size_mm': self.pixel_size_mm,
            'limbs': limbs_dicts
        }
        return frame_dict


    def mean_probability(self):
        if not self.limbs:
            return 0.0
        total_probability = sum(limb.probability for limb in self.limbs)
        return total_probability / len(self.limbs)


    def hasLimb(self, limb_name):
        """
        check if the Frame contains a limb with the specified name.

        :param limb_name: Name of the limb to check
        :return: True if the limb is found, False otherwise
        """
        return any((limb.name == limb_name and limb.probability >= 0.5) for limb in self.limbs)

    def getLimb(self, limb_name):
        """
        Get the LimbData object for the specified limb name.

        :param limb_name: Name of the limb to retrieve
        :return: LimbData object if found, otherwise None
        """
        if self.hasLimb(limb_name):
            return next(limb for limb in self.limbs if limb.name == limb_name)
        return None


    def check_overlap(self, limb_name1, limb_name2, tolerance_mm):
        """
        Check if the two specified limbs overlap within the given tolerance.

        :param limb_name1: Name of the first limb
        :param limb_name2: Name of the second limb
        :param tolerance_mm: Overlap tolerance in millimeters
        :return: Frame number if the limbs overlap, otherwise None
        """
        if not self.pixel_size_mm:
            return None
            
        if self.hasLimb(limb_name1) and self.hasLimb(limb_name2):
            limb1 = next(limb for limb in self.limbs if limb.name == limb_name1)
            limb2 = next(limb for limb in self.limbs if limb.name == limb_name2)
            
            # Calculate the distance between the two limbs in pixels
            distance_px = math.sqrt((limb1.x - limb2.x) ** 2 + (limb1.y - limb2.y) ** 2)
            # Convert the distance to millimeters
            
            distance_mm = distance_px * self.pixel_size_mm
            
            # Check if the distance is within the tolerance
            if distance_mm < tolerance_mm:
                return self.frame_number
        return None


class Clip:
    def __init__(self, frames, frame_rate=30):
        """
        Initialize the Clip object.

        :param frames: List of Frame objects
        :param frame_rate: Frame rate (default is 30)
        """
        self.frames = frames  # List of Frame objects
        self.frame_rate = frame_rate  # Frame rate

    def check_overlap(self, limb_name1, limb_name2, tolerance_mm):
        """
        Check for overlaps between the specified limbs in all frames.

        :param limb_name1: Name of the first limb
        :param limb_name2: Name of the second limb
        :param tolerance_mm: Overlap tolerance in millimeters
        :return: List of frame numbers with overlap, or None if no overlap is found
        """
        overlap_frames = []

        for frame in self.frames:
            overlap_frame_number = frame.check_overlap(limb_name1, limb_name2, tolerance_mm)
            if overlap_frame_number is not None:
                overlap_frames.append(overlap_frame_number)

        return overlap_frames if overlap_frames else None


    def getAvgVel(self, limb_name, frame_numbers):
        """
        Calculate the average velocity of a limb over multiple frames.

        :param limb_name: Name of the limb
        :param frame_numbers: List of frame numbers
        :return: Average velocity in cm/s
        """
        if len(frame_numbers) < 2:
            return 0.0
        
        total_distance_mm = 0.0
        total_time_s = 0.0
        
        for i in range(len(frame_numbers) - 1):
            frame_num1 = frame_numbers[i]
            frame_num2 = frame_numbers[i + 1]
            
            frame1 = next((frame for frame in self.frames if frame.frame_number == frame_num1), None)
            frame2 = next((frame for frame in self.frames if frame.frame_number == frame_num2), None)
            
            if frame1 is None or frame2 is None:
                continue
            
            limb1 = frame1.getLimb(limb_name)
            limb2 = frame2.getLimb(limb_name)
            
            if limb1 is None or limb2 is None:
                continue

            distance_px = math.sqrt((limb1.x - limb2.x) ** 2 + (limb1.y - limb2.y) ** 2)
            
            if frame1.pixel_size_mm is None and frame2.pixel_size_mm is None:
                continue
            elif frame1.pixel_size_mm is None:
                avg_pixel_size_mm = frame2.pixel_size_mm
            elif frame2.pixel_size_mm is None:
                avg_pixel_size_mm = frame1.pixel_size_mm
            else:
                avg_pixel_size_mm = (frame1.pixel_size_mm + frame2.pixel_size_mm) / 2
            
            distance_mm = distance_px * avg_pixel_size_mm
            
            time_s = abs(frame_num2 - frame_num1) / self.frame_rate
            
            total_distance_mm += distance_mm
            total_time_s += time_s
        
        if total_time_s == 0:
            return 0.0
        
        avg_velocity_cm_s = (total_distance_mm / 10) / total_time_s
        
        return avg_velocity_cm_s


    def calculateDirection(self, limb_name, frame_num1, frame_num2):
        """
        Calculate the direction of a limb's movement between two frames.

        :param limb_name: Name of the limb
        :param frame_num1: First frame number
        :param frame_num2: Second frame number
        :return: Direction in degrees, or None if calculation is not possible
        """
        if frame_num1 < 0 or frame_num2 < 0:
            return None
        
        frame1 = next((frame for frame in self.frames if frame.frame_number == frame_num1), None)
        frame2 = next((frame for frame in self.frames if frame.frame_number == frame_num2), None)
        
        if frame1 is None or frame2 is None:
            return None
        
        if not frame1.hasLimb(limb_name) or not frame2.hasLimb(limb_name):
            return None
        
        limb1 = frame1.getLimb(limb_name)
        limb2 = frame2.getLimb(limb_name)
        
        delta_x = limb2.x - limb1.x
        delta_y = limb2.y - limb1.y
        
        direction_rad = math.atan2(delta_y, delta_x)
        direction_deg = math.degrees(direction_rad)
        
        return direction_deg


    def plot_polar_directions(self, limb_name, frame_numbers):
        """
        Plot the direction of a limb's movement over a series of frames using a polar plot.

        :param limb_name: Name of the limb
        :param frame_numbers: List of frame numbers to calculate direction between
        """
        directions = []
        frame_pairs = zip(frame_numbers[:-1], frame_numbers[1:])
        
        for frame_num1, frame_num2 in frame_pairs:
            direction = self.calculateDirection(limb_name, frame_num1, frame_num2)
            if direction is not None:
                directions.append(math.radians(direction))
            else:
                directions.append(float('nan'))  # Use NaN for invalid direction
        
        fig = plt.figure(figsize=(10, 6))
        ax = fig.add_subplot(111, polar=True)
        ax.plot(directions, range(len(directions)), marker='o', linestyle='-')
        ax.set_theta_direction(-1)
        ax.set_theta_zero_location('N')
        ax.set_rlabel_position(180)
        plt.title(f'Polar Plot of {limb_name} Movement Directions')
        plt.show


    def plot_quiver_directions(self, limb_name, frame_numbers):
        """
        Plot the direction of a limb's movement over a series of frames using a quiver plot.

        :param limb_name: Name of the limb
        :param frame_numbers: List of frame numbers to calculate direction between
        """
        X = []
        Y = []
        U = []
        V = []

        for frame_num1, frame_num2 in zip(frame_numbers[:-1], frame_numbers[1:]):
            frame1 = next((frame for frame in self.frames if frame.frame_number == frame_num1), None)
            frame2 = next((frame for frame in self.frames if frame.frame_number == frame_num2), None)
            
            if frame1 is None or frame2 is None:
                continue
            
            limb1 = frame1.getLimb(limb_name)
            limb2 = frame2.getLimb(limb_name)
            
            if limb1 is None or limb2 is None:
                continue
            
            X.append(limb1.x)
            Y.append(limb1.y)
            U.append(limb2.x - limb1.x)
            V.append(limb2.y - limb1.y)

        fig, ax = plt.subplots()
        ax.quiver(X, Y, U, V, angles='xy', scale_units='xy', scale=1)
        ax.set_aspect('equal')
        plt.title(f'Quiver Plot of {limb_name} Movement')
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.grid(True)
        plt.show()

    # def detect_freezing(self, limb_names, frame_range, tolerance_mm, min_frames):
    #     freezing_segments = []
    #     current_segment = []

    #     for frame in self.frames:
    #         if frame.frame_number < frame_range[0] or frame.frame_number > frame_range[1]:
    #             continue

    #         all_stationary = True

    #         for limb_name in limb_names:
    #             limb = frame.getLimb(limb_name)
    #             if limb is None:
    #                 all_stationary = False
    #                 break

    #             prev_frame = next((f for f in self.frames if f.frame_number == frame.frame_number - 1), None)
    #             if prev_frame:
    #                 prev_limb = prev_frame.getLimb(limb_name)
    #                 if prev_limb:
    #                     distance_px = math.sqrt((limb.x - prev_limb.x) ** 2 + (limb.y - prev_limb.y) ** 2)
    #                     if frame.pixel_size_mm is None and prev_frame.pixel_size_mm is None:
    #                         all_stationary = False
    #                         break
    #                     elif frame.pixel_size_mm is None:
    #                         pixel_size_mm = prev_frame.pixel_size_mm
    #                     elif prev_frame.pixel_size_mm is None:
    #                         pixel_size_mm = frame.pixel_size_mm
    #                     else:
    #                         pixel_size_mm = (frame.pixel_size_mm + prev_frame.pixel_size_mm) / 2

    #                     distance_mm = distance_px * pixel_size_mm
    #                     if distance_mm > tolerance_mm:
    #                         all_stationary = False
    #                         break

    #         if all_stationary:
    #             current_segment.append(frame.frame_number)
    #         else:
    #             if len(current_segment) >= min_frames:
    #                 freezing_segments.append((current_segment[0], current_segment[-1]))
    #             current_segment = []

    #     if len(current_segment) >= min_frames:
    #         freezing_segments.append((current_segment[0], current_segment[-1]))
        
    #     return freezing_segments

    def detect_freezing(self, limb_names, frame_range, tolerance_mm, max_tolerance_mm, min_frames):
        freezing_segments = []
        current_segment = []
        segment_start_frame = None
        limb_positions = {}

        for frame in self.frames:
            if frame.frame_number < frame_range[0] or frame.frame_number > frame_range[1]:
                continue

            all_stationary = True
            if segment_start_frame is None:
                segment_start_frame = frame.frame_number
                limb_positions = {limb_name: [] for limb_name in limb_names}

            for limb_name in limb_names:
                limb = frame.getLimb(limb_name)
                if limb is None:
                    all_stationary = False
                    break
                
                prev_frame = next((f for f in self.frames if f.frame_number == frame.frame_number - 1), None)
                if prev_frame:
                    prev_limb = prev_frame.getLimb(limb_name)
                    if prev_limb:
                        distance_px = math.sqrt((limb.x - prev_limb.x) ** 2 + (limb.y - prev_limb.y) ** 2)
                        pixel_size_mm = frame.pixel_size_mm if frame.pixel_size_mm else prev_frame.pixel_size_mm
                        distance_mm = distance_px * pixel_size_mm
                        if distance_mm > tolerance_mm:
                            all_stationary = False
                            break
                
                limb_positions[limb_name].append((limb.x, limb.y))
                
                if len(limb_positions[limb_name]) > 1:
                    max_distance = max(
                        math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)
                        for i, p1 in enumerate(limb_positions[limb_name])
                        for p2 in limb_positions[limb_name][i+1:]
                    )
                    
                    if frame.pixel_size_mm is None:
                        all_stationary = False
                        break
                    
                    distance_mm = max_distance * frame.pixel_size_mm
                    if distance_mm > max_tolerance_mm:
                        all_stationary = False
                        break

            if all_stationary:
                current_segment.append(frame.frame_number)
            else:
                if len(current_segment) >= min_frames:
                    freezing_segments.append((current_segment[0], current_segment[-1]))
                current_segment = []
                segment_start_frame = None
                limb_positions = {}

        if len(current_segment) >= min_frames:
            freezing_segments.append((current_segment[0], current_segment[-1]))
        
        return freezing_segments



class Experiment:
    def __init__(self, GCaMP_data, Dlc_annotations, Mouse_id, Genotype, Sex, Date_of_birth, construct, Region_of_interest, paradigm, grooming_annotations=None, headipping_annotations=None, freezing_annotations=None):
        self.GCaMP_data = GCaMP_data  # Instance of GCaMPData class
        self.Dlc_annotations = Dlc_annotations  # DLC data (likely a DataFrame or array)
        self.Mouse_id = Mouse_id  # Unique identifier for the mouse
        self.Genotype = Genotype  # Mouse genotype (e.g., WT, KO)
        self.Sex = Sex  # Male or Female
        self.Date_of_birth = Date_of_birth  # Birth date as a `datetime.date`
        self.construct = construct  # String representing construct (e.g., GCaMP6s)
        self.Region_of_interest = Region_of_interest  # Brain region (e.g., CeA)
        self.paradigm = paradigm  # Experimental paradigm (e.g., optogenetics, behavioral task)

        # Replacing behavioral_annotations with behavioral_annotations
        self.behavioral_annotations = {
            "Grooming_annotations": grooming_annotations,
            "Headipping_annotations": headipping_annotations,
            "Freezing_annotations": freezing_annotations
        }

        # Initialize smooth_grooming_bouts as None
        self.smooth_grooming_bouts: Optional[List[Tuple[int, int]]] = None

    @property
    def Grooming_annotations(self):
        return self.behavioral_annotations["Grooming_annotations"]

    @property
    def Headipping_annotations(self):
        return self.behavioral_annotations["Headipping_annotations"]
    
    @property
    def Freezing_annotations(self):
        return self.behavioral_annotations["Freezing_annotations"]
    
    @property
    def grooming_annotations_count(self):
        """
        Count the number of grooming bouts in Grooming_annotations.
        """
        if self.Grooming_annotations:
            return len(self.Grooming_annotations)
        return 0

    @property
    def smooth_grooming_bouts_count(self):
        """
        Count the number of grooming bouts in smooth_grooming_bouts.
        """
        if self.smooth_grooming_bouts:
            return len(self.smooth_grooming_bouts)
        return 0

    def smooth_grooming_bouts_generator(self, threshold: int = 60):
        """
        Generate smooth grooming bouts by merging adjacent bouts 
        if the gap between them is less than the specified threshold.

        Parameters:
            threshold: The maximum gap (in frames) between grooming bouts to consider them adjacent.

        Updates:
            self.smooth_grooming_bouts: List of tuples with merged grooming bouts.
        """
        if not self.Grooming_annotations:
            self.smooth_grooming_bouts = None
            return

        smoothed_bouts = []
        sorted_bouts = sorted(self.Grooming_annotations)  # Ensure the bouts are sorted
        current_bout = sorted_bouts[0]  # Start with the first bout

        for next_bout in sorted_bouts[1:]:
            # Check if the current bout and next bout are adjacent
            if next_bout[0] - current_bout[1] <= threshold:
                # Merge the current and next bouts
                current_bout = (current_bout[0], next_bout[1])
            else:
                # Add the current bout to the smoothed list and move to the next one
                smoothed_bouts.append(current_bout)
                current_bout = next_bout

        # Add the last processed bout
        smoothed_bouts.append(current_bout)

        self.smooth_grooming_bouts = smoothed_bouts

    def __repr__(self):
        return (f"Experiment(Mouse_id={self.Mouse_id}, Genotype={self.Genotype}, Sex={self.Sex}, "
                f"Date_of_birth={self.Date_of_birth}, construct={self.construct}, "
                f"Region_of_interest={self.Region_of_interest}, paradigm={self.paradigm})")

    def to_dict(self):
        return {
            "GCaMP_data": self.GCaMP_data.to_dict() if self.GCaMP_data else None,
            "Dlc_annotations": self.Dlc_annotations,
            "behavioral_annotations": self.behavioral_annotations,
            "Mouse_id": self.Mouse_id,
            "Genotype": self.Genotype,
            "Sex": self.Sex,
            "Date_of_birth": self.Date_of_birth,
            "construct": self.construct,
            "Region_of_interest": self.Region_of_interest,
            "paradigm": self.paradigm,
            "smooth_grooming_bouts": self.smooth_grooming_bouts
        }


    def get_gcamp_data(self, start_frame, end_frame, extra_time_before=0, extra_time_after=0):
        """
        Get GCaMP_dF_F and time_seconds arrays for the specified frame range.
        
        :param start_frame: The starting frame in 30fps.
        :param end_frame: The ending frame in 30fps.
        :param extra_time_before: Extra time in seconds before the start (default is 0).
        :param extra_time_after: Extra time in seconds after the end (default is 0).
        :return: Corresponding GCaMP_dF_F and time_seconds arrays.
        """

        # Convert start and end frames to seconds (30 fps)
        start_time = start_frame / 30.0
        end_time = end_frame / 30.0
        
        # Adjust with extra time before and after
        start_time -= extra_time_before
        end_time += extra_time_after

        # Ensure the times stay within the valid range of GCaMP time
        start_time = max(0, start_time)
        end_time = min(self.GCaMP_data.time[-1], end_time)

        # Convert the start and end times to indices in the GCaMP data based on the sample rate
        start_idx = (self.GCaMP_data.time >= start_time).argmax()
        end_idx = (self.GCaMP_data.time <= end_time).sum()  # Last index where time <= end_time
        
        # Get the corresponding GCaMP_dF_F and time arrays
        gcamp_dF_F = self.GCaMP_data.GCaMP_dF_F[start_idx:end_idx+1]
        time_seconds = self.GCaMP_data.time[start_idx:end_idx+1]

        return gcamp_dF_F, time_seconds


    def get_gcamp_data_Z_score(self, start_frame, end_frame, extra_time_before=0, extra_time_after=0):
        """
        Get GCaMP_dF_F and time_seconds arrays for the specified frame range.
        
        :param start_frame: The starting frame in 30fps.
        :param end_frame: The ending frame in 30fps.
        :param extra_time_before: Extra time in seconds before the start (default is 0).
        :param extra_time_after: Extra time in seconds after the end (default is 0).
        :return: Corresponding GCaMP_dF_F and time_seconds arrays.
        """

        # Convert start and end frames to seconds (30 fps)
        start_time = start_frame / 30.0
        end_time = end_frame / 30.0
        
        # Adjust with extra time before and after
        start_time -= extra_time_before
        end_time += extra_time_after

        # Ensure the times stay within the valid range of GCaMP time
        start_time = max(0, start_time)
        end_time = min(self.GCaMP_data.time[-1], end_time)

        # Convert the start and end times to indices in the GCaMP data based on the sample rate
        start_idx = (self.GCaMP_data.time >= start_time).argmax()
        end_idx = (self.GCaMP_data.time <= end_time).sum()  # Last index where time <= end_time

        # Get the corresponding GCaMP_Zscore and time arrays
        gcamp_z_score = self.GCaMP_data.z_score[start_idx:end_idx+1]
        time_seconds = self.GCaMP_data.time[start_idx:end_idx+1]

        return gcamp_z_score, time_seconds
    

    def get_gcamp_data_for_annotations(self, categories=None, n=None, extra_time_before=0, extra_time_after=0):
        """
        Get GCaMP_dF_F and time_seconds arrays for specified behavioral_annotations categories.

        :param categories: A list of annotation categories (e.g., ['Grooming_annotations', 'Food_annotations']).
                        If None, processes all categories in `behavioral_annotations`.
        :param n: Number of bouts to process per category (default is None, meaning all).
        :param extra_time_before: Extra time in seconds before each bout (default is 0).
        :param extra_time_after: Extra time in seconds after each bout (default is 0).
        :return: A dictionary with category names as keys and a list of tuples (GCaMP_dF_F, time_seconds) as values.
        """
        if categories is None:
            categories = list(self.behavioral_annotations.keys())  # Default to all categories
        
        invalid_categories = [cat for cat in categories if cat not in self.behavioral_annotations]
        if invalid_categories:
            raise ValueError(f"Invalid categories: {invalid_categories}. Must be one of {list(self.behavioral_annotations.keys())}.")
        
        results = {}
        for category in categories:
            bouts = (self.behavioral_annotations[category][:n] if n is not None else self.behavioral_annotations[category]) if self.behavioral_annotations.get(category) is not None else []
            
            category_results = []
            for start_frame, end_frame in bouts if bouts is not None else []:
                gcamp_dF_F, time_seconds = self.get_gcamp_data(
                    start_frame, end_frame, 
                    extra_time_before=extra_time_before, 
                    extra_time_after=extra_time_after
                )
                category_results.append((gcamp_dF_F, time_seconds))
            
            results[category] = category_results
        return results


    def get_gcamp_z_score_for_annotations(self, category, n=None, extra_time_before=0, extra_time_after=0):
        """
        Get GCaMP_dF_F and time_seconds arrays for behavioral annotation bouts.

        :param n: Number of bouts to process (default is None, meaning all).
        :param extra_time_before: Extra time in seconds before each bout (default is 0).
        :param extra_time_after: Extra time in seconds after each bout (default is 0).
        :return: A list of tuples with (GCaMP_dF_F, time_seconds) for each bout.
        """
        
        if category not in self.behavioral_annotations:
            raise ValueError(f"Invalid category: {category}. Must be one of {list(self.behavioral_annotations.keys())}.")
        
        bouts = self.behavioral_annotations[category][:n] if n is not None else self.behavioral_annotations[category]

        results = []

        # Iterate over the behavioral bouts
        for start_frame, end_frame in bouts if bouts is not None else []:
            # Call the get_gcamp_data_Z_score function for each pair of start and end frames
            gcamp_z_score, time_seconds = self.get_gcamp_data_Z_score(start_frame, end_frame, 
                                                           extra_time_before=extra_time_before, 
                                                           extra_time_after=extra_time_after)
            # Append the result as a tuple to the results list
            results.append((gcamp_z_score, time_seconds))
        return results
    
    def shade_event(self, start_time, end_time, color='gray', alpha=0.5, linestyle='-', linewidth=1.0):
        # Plot vertical lines for start and end times
        plt.axvline(start_time, color=color, linestyle=linestyle, linewidth=linewidth)
        plt.axvline(end_time, color=color, linestyle=linestyle, linewidth=linewidth)
        
        # Shade the region between start and end times
        plt.fill_betweenx([-1100, 1100], start_time, end_time, color=color, alpha=alpha)
    


    def plot_gcamp_with_annotations(self, categories=None, xlim_start=None, xlim_end=None, ylim_start=None, ylim_end=None, alpha=0.3, linestyle='-', linewidth=0, show_legend=False):
        """
        Plot GCaMP data with shaded events for one or more annotation categories.

        :param categories: A list of annotation categories to plot (default is all categories in behavioral_annotations).
        :param xlim_start: Start of the x-axis range.
        :param xlim_end: End of the x-axis range.
        :param ylim_start: Start of the y-axis range.
        :param ylim_end: End of the y-axis range.
        :param alpha: Transparency of the shading.
        :param linestyle: Line style for the shading.
        :param linewidth: Line width for the shading.
        :param show_legend: Whether to show the legend (default is True).
        """
        if categories is None:
            categories = list(self.behavioral_annotations.keys())  # Default to all categories

        invalid_categories = [cat for cat in categories if cat not in self.behavioral_annotations]
        if invalid_categories:
            raise ValueError(f"Invalid categories: {invalid_categories}. Must be one of {list(self.behavioral_annotations.keys())}.")

        # Define a unique color for each category
        color_palette = {
            "Grooming_annotations": 'blue',
            "Headipping_annotations": 'brown',
            "Freezing_annotations": 'red'
        }

        gcamp_dF_F = self.GCaMP_data.GCaMP_dF_F
        time_seconds = self.GCaMP_data.time

        plt.figure(figsize=[14, 6])
        plt.plot(time_seconds, gcamp_dF_F, label='GCaMP dF/F', color='green')

        # Plot events for each category with unique colors
        for category in categories:
            if category not in self.behavioral_annotations or category not in color_palette:
                continue
            color = color_palette[category]
            if self.behavioral_annotations.get(category) is not None:  # Check if the category exists and is not None
                for start_frame, end_frame in self.behavioral_annotations[category]:
                    start_time = start_frame / 30  # Convert to seconds
                    end_time = end_frame / 30  # Convert to seconds
                    plt.axvline(start_time, color=color, linestyle=linestyle, linewidth=linewidth)
                    plt.axvline(end_time, color=color, linestyle=linestyle, linewidth=linewidth)
                    plt.fill_betweenx([-1000, 1000], start_time, end_time, color=color, alpha=alpha)

        xlim_start = xlim_start or min(time_seconds)
        xlim_end = xlim_end or max(time_seconds)
        ylim_start = ylim_start or min(gcamp_dF_F)
        ylim_end = ylim_end or max(gcamp_dF_F)
        
        plt.xlim(xlim_start, xlim_end)
        plt.ylim(ylim_start, ylim_end)
        plt.xlabel('Time (seconds)', fontsize=20)
        plt.ylabel('GCaMP dF/F (%)', fontsize=20)
        plt.title('GCaMP dF/F with Behavioral Annotations', fontsize=26)
        plt.xticks(fontsize=22)  # Adjust fontsize for x ticks
        plt.yticks(fontsize=22)  # Adjust fontsize for y ticks
        plt.axhline(0, color="gray", linestyle="--")
        
        if show_legend:
            # Add a custom legend for each category
            handles = [plt.Line2D([0], [0], color=color_palette[cat], lw=4, label=cat) for cat in categories if cat in color_palette]
            plt.legend(handles=handles)
        
        plt.show()


    def plot_gcamp_zscore_with_annotations(self, categories=None, xlim_start=None, xlim_end=None, ylim_start=None, ylim_end=None, color='b', alpha=0.3, linestyle='-', linewidth=0):
        """
        Plot GCaMP data with shaded behavioral events.
        """
        if categories is None:
            categories = list(self.behavioral_annotations.keys())  # Default to all categories

        # Define a unique color for each category
        color_palette = {
            "Grooming_annotations": 'blue',
            "Headipping_annotations": 'brown',
            "Freezing_annotations": 'red'
        }

        # Get the GCaMP_dF_F and time data
        Gcamp_zscore = self.GCaMP_data.z_score
        time_seconds = self.GCaMP_data.time

        # Make default figure size larger
        plt.figure(figsize=[14, 6])

        # Plot the GCaMP_dF_F trace
        plt.plot(time_seconds, Gcamp_zscore, label='GCaMP Zscore', color='green')

        # Iterate over behavioral annotations to shade events
        # Plot events for each category with unique colors
        for category in categories:
            if category not in self.behavioral_annotations or category not in color_palette:
                continue
            color = color_palette[category]
            if self.behavioral_annotations.get(category) is not None:
                for start_frame, end_frame in self.behavioral_annotations[category]:
                    start_time = start_frame / 30  # Convert to seconds
                    end_time = end_frame / 30  # Convert to seconds
                    plt.axvline(start_time, color=color, linestyle=linestyle, linewidth=linewidth)
                    plt.axvline(end_time, color=color, linestyle=linestyle, linewidth=linewidth)
                    plt.fill_betweenx([-1000, 1000], start_time, end_time, color=color, alpha=alpha)

         # Set x-axis limits based on provided arguments or default to min/max of time data
        if xlim_start is None:
            xlim_start = min(time_seconds)
        if xlim_end is None:
            xlim_end = max(time_seconds)
         
         # Set y-axis limits based on provided arguments or default to min/max of time data
        if ylim_start is None:
            ylim_start = min(Gcamp_zscore)
        if ylim_end is None:
            ylim_end = max(Gcamp_zscore)
        
        # Set limits and labels for the plot
        plt.xlim(xlim_start, xlim_end)  # Set x-axis limits
        # Set limits and labels for the plot
        plt.ylim(ylim_start, ylim_end)  # Set y-axis limits

        # Calculate y-axis limits
        gcamp_min = min(Gcamp_zscore)
        gcamp_max = max(Gcamp_zscore)
        y_range = gcamp_max - gcamp_min
        adjustment = 0.02 * abs(y_range)  # 5% of the range

        plt.ylim(ylim_start - adjustment, ylim_end + adjustment)
    
        plt.xlabel('Time (seconds)', fontsize=26)
        plt.ylabel('GCaMP Z score (%)', fontsize=26)
        plt.title('GCaMP Z score', fontsize=30)
        plt.xticks(fontsize=22)  # Adjust fontsize for x ticks
        plt.yticks(fontsize=22)  # Adjust fontsize for y ticks
        # plt.grid(True)  # Uncomment to enable grid
        plt.axhline(0, color="gray", linestyle="--")

        plt.legend()
        plt.show()

    def get_gcamp_data_for_smooth_grooming_bouts(self, categories=None, xlim_start=None, xlim_end=None, ylim_start=None, ylim_end=None, color='b', alpha=0.3, linestyle='-', linewidth=0):
        """
        Get GCaMP_dF_F and time_seconds arrays for each smooth grooming bout.
        
        :param n: Number of bouts to process (default is None, meaning all).
        :param extra_time_before: Extra time in seconds before each bout (default is 0).
        :param extra_time_after: Extra time in seconds after each bout (default is 0).
        :return: A list of tuples (GCaMP_dF_F, time_seconds) for each smooth bout.
        """
        bouts = self.smooth_grooming_bouts
        if bouts is None:
            return []
        if n is not None:
            bouts = bouts[:n]
        
        results = []
        for start_frame, end_frame in bouts:
            gcamp_dF_F, time_seconds = self.get_gcamp_data(start_frame, end_frame, extra_time_before, extra_time_after)
            results.append((gcamp_dF_F, time_seconds))
        return results

    def get_gcamp_z_score_for_smooth_grooming_bouts(self, extra_time_before=0, extra_time_after=0, 
                                                    categories=None, xlim_start=None, xlim_end=None, 
                                                    ylim_start=None, ylim_end=None, color='b', alpha=0.3, 
                                                    linestyle='-', linewidth=0):
        """
        Get GCaMP_z_score and time_seconds arrays for each smooth grooming bout.
        Allows extra time before/after the bout.
        """
        bouts = self.smooth_grooming_bouts
        if bouts is None:
            return []
        # If you need to support an "n" or limit the bouts, you could add that too.
        results = []
        for start_frame, end_frame in bouts:
            # Now pass extra_time_before and extra_time_after to get_gcamp_data_Z_score.
            gcamp_z_score, time_seconds = self.get_gcamp_data_Z_score(start_frame, end_frame, 
                                                                    extra_time_before=extra_time_before, 
                                                                    extra_time_after=extra_time_after)
            results.append((gcamp_z_score, time_seconds))
        return results


    def plot_gcamp_with_smooth_grooming_bouts(self, xlim_start=None, xlim_end=None, ylim_start=None, ylim_end=None, color='blue', alpha=0.3, linestyle='-', linewidth=0, show_legend=False):
        """
        Plot GCaMP dF/F data with smooth grooming bouts shaded on the plot.
        
        :param xlim_start: Start of the x-axis range.
        :param xlim_end: End of the x-axis range.
        :param ylim_start: Start of the y-axis range.
        :param ylim_end: End of the y-axis range.
        :param color: Color to use for the smooth bouts shading.
        :param alpha: Transparency of the shading.
        :param linestyle: Line style for the bout boundaries.
        :param linewidth: Line width for the bout boundaries.
        :param show_legend: Whether to show the legend.
        """
        gcamp_dF_F = self.GCaMP_data.GCaMP_dF_F
        time_seconds = self.GCaMP_data.time

        plt.figure(figsize=[14, 6])
        plt.plot(time_seconds, gcamp_dF_F, label='GCaMP dF/F', color='green')

        if self.smooth_grooming_bouts:
            for start_frame, end_frame in self.smooth_grooming_bouts:
                start_time = start_frame / 30  # Convert frames to seconds
                end_time = end_frame / 30
                plt.axvline(start_time, color=color, linestyle=linestyle, linewidth=linewidth)
                plt.axvline(end_time, color=color, linestyle=linestyle, linewidth=linewidth)
                plt.fill_betweenx([-1000, 1000], start_time, end_time, color=color, alpha=alpha)

        xlim_start = xlim_start or min(time_seconds)
        xlim_end = xlim_end or max(time_seconds)
        ylim_start = ylim_start or min(gcamp_dF_F)
        ylim_end = ylim_end or max(gcamp_dF_F)

        plt.xlim(xlim_start, xlim_end)
        plt.ylim(ylim_start, ylim_end)
        plt.xlabel('Time (seconds)', fontsize=20)
        plt.ylabel('GCaMP dF/F (%)', fontsize=20)
        plt.title('GCaMP dF/F with Smooth Grooming Bouts', fontsize=26)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.axhline(0, color="gray", linestyle="--")

        if show_legend:
            plt.legend()

        plt.show()

    def plot_gcamp_zscore_with_smooth_grooming_bouts(self, xlim_start=None, xlim_end=None, ylim_start=None, ylim_end=None, color='blue', alpha=0.3, linestyle='-', linewidth=0, show_legend=False):
        """
        Plot GCaMP Z score data with smooth grooming bouts shaded on the plot.
        
        :param xlim_start: Start of the x-axis range.
        :param xlim_end: End of the x-axis range.
        :param ylim_start: Start of the y-axis range.
        :param ylim_end: End of the y-axis range.
        :param color: Color to use for the smooth bouts shading.
        :param alpha: Transparency of the shading.
        :param linestyle: Line style for the bout boundaries.
        :param linewidth: Line width for the bout boundaries.
        :param show_legend: Whether to show the legend.
        """
        # Extract GCaMP data and time (assuming these are available in your object)
        Gcamp_zscore = self.GCaMP_data.z_score
        time_seconds = self.GCaMP_data.time

        # Set up the figure
        plt.figure(figsize=[14, 6])
        
        # Plot the GCaMP Z-score trace
        plt.plot(time_seconds, Gcamp_zscore, label='GCaMP Zscore', color='green')

        # Plot smooth grooming bouts, converting frame indices to seconds (assuming 30 fps)
        if self.smooth_grooming_bouts:
            for start_frame, end_frame in self.smooth_grooming_bouts:
                start_time = start_frame / 30
                end_time = end_frame / 30
                plt.axvline(start_time, color=color, linestyle=linestyle, linewidth=linewidth)
                plt.axvline(end_time, color=color, linestyle=linestyle, linewidth=linewidth)
                plt.fill_betweenx([-1000, 1000], start_time, end_time, color=color, alpha=alpha)

        # Determine x-axis limits
        if xlim_start is None:
            xlim_start = min(time_seconds)
        if xlim_end is None:
            xlim_end = max(time_seconds)
        plt.xlim(xlim_start, xlim_end)

        # Determine y-axis limits from the data if not provided
        if ylim_start is None:
            ylim_start = min(Gcamp_zscore)
        if ylim_end is None:
            ylim_end = max(Gcamp_zscore)
        
        # Adjust y-axis limits to add a margin (here 2% of the range)
        gcamp_min = min(Gcamp_zscore)
        gcamp_max = max(Gcamp_zscore)
        y_range = gcamp_max - gcamp_min
        adjustment = 0.02 * abs(y_range)
        plt.ylim(ylim_start - adjustment, ylim_end + adjustment)

        # Add labels, title, and adjust ticks
        plt.xlabel('Time (seconds)', fontsize=26)
        plt.ylabel('GCaMP Z score', fontsize=26)
        plt.title('GCaMP Z score with Smooth Grooming Bouts', fontsize=30)
        plt.xticks(fontsize=22)
        plt.yticks(fontsize=22)
        plt.axhline(0, color="gray", linestyle="--")
        
        if show_legend:
            plt.legend()
        
        plt.show()

    def animate_gcamp_with_behavioral_annotations(self, start_frame, end_frame, video_fps=10, video_duration=10, output_path="gcamp_animation.mp4", extra_time_before=0, extra_time_after=0, annotations=None, y_limits=None):
        """
        Animate a subset of GCaMP data with behavioral events and save as an MP4 video.

        :param start_frame: The starting frame (in 30 fps).
        :param end_frame: The ending frame (in 30 fps).
        :param video_fps: Frames per second for the video.
        :param video_duration: Duration of the video in seconds.
        :param output_path: Path to save the MP4 video.
        :param extra_time_before: Extra time in seconds before the start (default is 0).
        :param extra_time_after: Extra time in seconds after the end (default is 0).
        :param annotations: Behavioral annotations as a list of tuples.
        :param y_limits: Optional tuple (ymin, ymax) to manually set the y-axis limits. If None, automatic scaling is used.
        """
        # Get data from the experiment using the provided frame range
        gcamp_dF_F, time_seconds = self.get_gcamp_data(start_frame, end_frame, extra_time_before, extra_time_after)
        annotations = annotations
        # Filter behavioral annotations to only include those within the start and end frame range
        filtered_annotations = [
            (start, end) for start, end in annotations
            if (start <= end_frame and end >= start_frame)  # Only include events within the specified range
        ]

        # Calculate total frames and step size
        total_frames = video_fps * video_duration
        step_size = len(time_seconds) // total_frames

        # Create figure and axis
        fig, ax = plt.subplots(figsize=(14, 6))
        ax.set_xlim(time_seconds[0], time_seconds[-1])
        
        # Use manual y-axis limits if provided; otherwise, use automatic scaling
        if y_limits is not None:
            ax.set_ylim(y_limits)
        else:
            ax.set_ylim(min(gcamp_dF_F) * 1.1, max(gcamp_dF_F) * 1.1)
            
        ax.set_xlabel("Time (seconds)", fontsize=22)
        ax.set_ylabel("GCaMP dF/F (%)", fontsize=22)
        ax.set_title("GCaMP8s Dynamics with Behavior", fontsize=26)
        plt.xticks(fontsize=18)  # Adjust fontsize for x ticks
        plt.yticks(fontsize=18)  # Adjust fontsize for y ticks

        # Initialize the line plot and shaded behavioral event patches
        line, = ax.plot([], [], label='GCaMP dF/F', color='#228B22', linewidth=3)
        shaded_patches = []

        # Store previously plotted data
        all_time = []
        all_gcamp = []

        # Update function for the animation
        def update(frame):
            # Calculate the current end index based on the frame
            start_idx = frame * step_size
            end_idx = min((frame + 1) * step_size, len(time_seconds))  # Ensure no out-of-bounds access

            # Only include data points for the current frame
            current_time = time_seconds[start_idx:end_idx]
            current_gcamp = gcamp_dF_F[start_idx:end_idx]

            # Update the stored data with the new points for this frame
            all_time.extend(current_time)
            all_gcamp.extend(current_gcamp)

            # Update the line plot with the previously and currently plotted data
            line.set_data(all_time, all_gcamp)

            # Clear and redraw shaded behavioral events progressively
            for patch in shaded_patches:
                patch.remove()
            shaded_patches.clear()

            for start_event, end_event in filtered_annotations:
                # Convert frames to seconds (assuming frame rate is 30 fps)
                start_time = start_event / 30
                end_time = end_event / 30

                # Only add event patches if the event start time is within the current frame's time range
                if start_time <= current_time[-1]:
                    patch = ax.axvspan(start_time, min(end_time, current_time[-1]),
                                    color='blue', alpha=0.3, linestyle='-', linewidth=0)
                    shaded_patches.append(patch)

            return [line] + shaded_patches

        # Create the animation
        ani = FuncAnimation(fig, update, frames=total_frames, blit=True)

        # Save the animation as an MP4 video
        writer = FFMpegWriter(fps=video_fps, metadata={"title": "GCaMP Animation"})
        ani.save(output_path, writer=writer)

        print(f"Animation saved to {output_path}")


# def animate_gcamp_with_behavioral_annotations(
#     self,
#     start_frame,
#     end_frame,
#     video_fps=10,
#     video_duration=10,
#     output_path="gcamp_animation.mp4",
#     extra_time_before=0,
#     extra_time_after=0,
#     annotations=None,                 # optional flat list of (start,end) for backward compatibility
#     y_limits=None,
#     categories=None,                  # e.g., ["Grooming_annotations"] to show only grooming
#     category_colors=None,             # dict mapping category -> color
#     show_category_lines=False         # draw start/end vertical lines per event
# ):
#     """
#     Animate a subset of GCaMP data with behavioral events (by category) and save as an MP4 video.

#     - If `categories` is provided, events are pulled from self.behavioral_annotations for those categories.
#     - If `categories` is None but self.behavioral_annotations exists, all categories found there are drawn.
#     - If `annotations` (flat list of tuples) is provided, those are drawn as a 'Custom' overlay.
#     """

#     # --- Get the GCaMP data/time for the requested window ---
#     gcamp_dF_F, time_seconds = self.get_gcamp_data(start_frame, end_frame, extra_time_before, extra_time_after)

#     # --- Build category -> events dict (events are lists of (start_frame, end_frame)) ---
#     events_by_category = {}

#     has_structured = hasattr(self, "behavioral_annotations") and isinstance(getattr(self, "behavioral_annotations"), dict)
#     if has_structured:
#         # Decide which categories to include
#         if categories is None:
#             # default to all categories in self.behavioral_annotations
#             categories_to_use = [c for c, ev in self.behavioral_annotations.items() if ev]
#         else:
#             categories_to_use = [c for c in categories if c in self.behavioral_annotations and self.behavioral_annotations[c]]

#         # Filter each category to only events overlapping [start_frame, end_frame]
#         for cat in categories_to_use:
#             raw_events = self.behavioral_annotations.get(cat, [])
#             filtered = [(s, e) for (s, e) in raw_events if (s <= end_frame and e >= start_frame)]
#             if filtered:
#                 events_by_category[cat] = filtered

#     # Optional flat annotations (back-compat)
#     custom_events = []
#     if annotations is not None and isinstance(annotations, (list, tuple)):
#         custom_events = [(s, e) for (s, e) in annotations if (s <= end_frame and e >= start_frame)]

#     # --- Colors per category ---
#     default_palette = {
#         "Grooming_annotations": "blue",
#         "Headipping_annotations": "brown",
#         "Freezing_annotations": "red",
#         # add more defaults if you have other categories
#     }
#     if category_colors is None:
#         category_colors = default_palette
#     else:
#         # merge with defaults so any missing category gets a sensible color
#         tmp = default_palette.copy()
#         tmp.update(category_colors)
#         category_colors = tmp

#     # --- Frame stepping logic ---
#     total_frames_requested = max(1, int(video_fps * video_duration))
#     if total_frames_requested >= len(time_seconds):
#         total_frames = len(time_seconds)
#         step_size = 1
#     else:
#         total_frames = total_frames_requested
#         step_size = max(1, len(time_seconds) // total_frames)

#     # --- Figure and axes ---
#     fig, ax = plt.subplots(figsize=(14, 6))
#     ax.set_xlim(time_seconds[0], time_seconds[-1])

#     if y_limits is not None:
#         ax.set_ylim(y_limits)
#     else:
#         ymin = min(gcamp_dF_F)
#         ymax = max(gcamp_dF_F)
#         margin = 0.1 * (ymax - ymin if ymax > ymin else (abs(ymax) + 1.0))
#         ax.set_ylim(ymin - margin, ymax + margin)

#     ax.set_xlabel("Time (seconds)", fontsize=22)
#     ax.set_ylabel("GCaMP dF/F (%)", fontsize=22)
#     ax.set_title("GCaMP8s Dynamics with Behavior", fontsize=26)
#     plt.xticks(fontsize=18)
#     plt.yticks(fontsize=18)

#     # --- Plot handles we update ---
#     (line,) = ax.plot([], [], label="GCaMP dF/F", color="#228B22", linewidth=3)

#     # For progressive shaded regions
#     shaded_patches = []
#     # Optional: per-category vertical lines
#     line_collections = []

#     # Keep legend entries unique
#     legend_done_for = set()

#     # Accumulate previously drawn data (progressive trace)
#     all_time = []
#     all_gcamp = []

#     # Helper to add a legend entry (only once per category/label)
#     def add_legend_once(label, handle):
#         if label not in legend_done_for:
#             legend_done_for.add(label)
#             # Matplotlib will pick these up in the final legend call

#     # --- Core update function per animation frame ---
#     def update(frame_idx):
#         start_idx = frame_idx * step_size
#         end_idx = min((frame_idx + 1) * step_size, len(time_seconds))

#         # Guard for empty slice (can happen at the very end)
#         if start_idx >= end_idx:
#             return [line] + shaded_patches + line_collections

#         current_time = time_seconds[start_idx:end_idx]
#         current_gcamp = gcamp_dF_F[start_idx:end_idx]

#         all_time.extend(current_time)
#         all_gcamp.extend(current_gcamp)
#         line.set_data(all_time, all_gcamp)

#         # Clear previous patches/lines
#         for patch in shaded_patches:
#             patch.remove()
#         shaded_patches.clear()

#         for lc in line_collections:
#             try:
#                 lc.remove()
#             except Exception:
#                 pass
#         line_collections.clear()

#         current_tmax = all_time[-1]  # up to where we are animating

#         # Draw category-based annotations
#         for cat, ev_list in events_by_category.items():
#             color = category_colors.get(cat, "gray")
#             for s_f, e_f in ev_list:
#                 s_t = s_f / 30.0
#                 e_t = e_f / 30.0
#                 if s_t <= current_tmax:
#                     # shaded span up to current time
#                     patch = ax.axvspan(s_t, min(e_t, current_tmax), color=color, alpha=0.30, linestyle="-", linewidth=0)
#                     shaded_patches.append(patch)
#                     if show_category_lines:
#                         v1 = ax.axvline(s_t, color=color, linestyle="--", linewidth=1.0)
#                         line_collections.append(v1)
#                         if e_t <= current_tmax:
#                             v2 = ax.axvline(e_t, color=color, linestyle="--", linewidth=1.0)
#                             line_collections.append(v2)
#             # add to legend once
#             add_legend_once(cat, patch if shaded_patches else line)

#         # Draw custom flat annotations (if any)
#         if custom_events:
#             for s_f, e_f in custom_events:
#                 s_t = s_f / 30.0
#                 e_t = e_f / 30.0
#                 if s_t <= current_tmax:
#                     patch = ax.axvspan(s_t, min(e_t, current_tmax), color="tab:blue", alpha=0.20, linestyle="-", linewidth=0)
#                     shaded_patches.append(patch)
#             add_legend_once("Custom", shaded_patches[-1] if shaded_patches else line)

#         # Update legend (once per frame is fine; entries deduped)
#         ax.legend(loc="upper right")

#         return [line] + shaded_patches + line_collections

#     # --- Create and save animation ---
#     ani = FuncAnimation(fig, update, frames=total_frames, blit=True)
#     writer = FFMpegWriter(fps=video_fps, metadata={"title": "GCaMP Animation"})
#     ani.save(output_path, writer=writer)
#     plt.close(fig)
#     print(f"Animation saved to {output_path}")
