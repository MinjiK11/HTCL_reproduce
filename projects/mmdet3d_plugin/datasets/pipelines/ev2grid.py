import numpy as np
import torch
from PIL import Image

class Event():
    """Load multi channel images from a list of separate channel files.

    Expects results['img_filename'] to be a list of filenames.

    Args:
        to_float32 (bool): Whether to convert the img to float32.
            Defaults to False.
        color_type (str): Color type of the file. Defaults to 'unchanged'.
    """

    def __init__(self,num_bins=3,norm_e=True):
        self.num_bins=3
        self.norm_e=True

    def fix_time(self,vect): ## vect : event의 시간들
        ref = np.ones(len(vect) - 1)
        y_hat = np.diff(vect) / ref ## y_hat : event간의 시간 차
        starts = np.where(y_hat < 0)[0] # y_hat 배열에서 첫번째로 0보다 작은 원소를 갖는 인덱스
        vect = np.asarray(vect)
        for i in range(len(starts)):
            vect[starts[i]+1:] += vect[starts[i]]
        return vect

    def ev2grid(self,events, width, height):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.

        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        :param device: device to use to perform computations
        :return voxel_grid: PyTorch event tensor (on the device specified)
        """

        num_bins = self.num_bins

        assert (events.shape[1] == 4)
        assert (num_bins > 0)
        assert (width > 0)
        assert (height > 0)

        with torch.no_grad():

            voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32).flatten()

            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events[-1, 0]
            first_stamp = events[0, 0]
            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT

            ts = events[:, 0]
            xs = events[:, 1].long() # event가 발생한 pixel x 좌표
            ys = events[:, 2].long() # event가 발생한 pixel y 좌표
            pols = events[:, 3].float() # polarity
            pols[pols == 0] = -1  # polarity should be +1 / -1

            tis = torch.floor(ts) # 각 요소보다 작거나 같은 가장 큰 정수로 이루어진 텐서 반환 (각 event가 어느 channel에 속하는지; channel은 정수이므로)
            tis_long = tis.long() 
            dts = ts - tis 
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            # each event contributes its polarity value to its two closest temporal channel
            valid_indices = tis < num_bins
            valid_indices &= tis >= 0
            voxel_grid.index_add_(dim=0,
                                    index=xs[valid_indices] + ys[valid_indices]
                                            * width  + tis_long[valid_indices] * height * width,
                                    source=vals_left[valid_indices])

            valid_indices = (tis + 1) < num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(dim=0,
                                    index=xs[valid_indices] + ys[valid_indices] * width
                                            + (tis_long[valid_indices] + 1) * height * width,
                                    source=vals_right[valid_indices])

            voxel_grid = voxel_grid.view(num_bins,height, width)

        return voxel_grid

    def ev2grid_binlast(self,events, width, height):
        """
        Build a voxel grid with bilinear interpolation in the time domain from a set of events.

        :param events: a [N x 4] NumPy array containing one event per row in the form: [timestamp, x, y, polarity]
        :param num_bins: number of bins in the temporal axis of the voxel grid
        :param width, height: dimensions of the voxel grid
        :param device: device to use to perform computations
        :return voxel_grid: PyTorch event tensor (on the device specified)
        """

        num_bins = self.num_bins

        assert (events.shape[1] == 4)
        assert (num_bins > 0)
        assert (width > 0)
        assert (height > 0)

        with torch.no_grad():

            voxel_grid = torch.zeros(num_bins, height, width, dtype=torch.float32).flatten()

            # normalize the event timestamps so that they lie between 0 and num_bins
            last_stamp = events[-1, 0]
            first_stamp = events[0, 0]
            deltaT = last_stamp - first_stamp

            if deltaT == 0:
                deltaT = 1.0

            events[:, 0] = (num_bins - 1) * (events[:, 0] - first_stamp) / deltaT

            ts = events[:, 0]
            xs = events[:, 1].long() # event가 발생한 pixel x 좌표
            ys = events[:, 2].long() # event가 발생한 pixel y 좌표
            pols = events[:, 3].float() # polarity
            pols[pols == 0] = -1  # polarity should be +1 / -1

            tis = torch.floor(ts) # 각 요소보다 작거나 같은 가장 큰 정수로 이루어진 텐서 반환 (각 event가 어느 channel에 속하는지; channel은 정수이므로)
            tis_long = tis.long() 
            dts = ts - tis 
            vals_left = pols * (1.0 - dts.float())
            vals_right = pols * dts.float()

            # each event contributes its polarity value to its two closest temporal channel
            valid_indices = tis < num_bins
            valid_indices &= tis >= 0
            voxel_grid.index_add_(dim=0,
                                    index=xs[valid_indices] * num_bins + ys[valid_indices]
                                            * width *num_bins  + tis_long[valid_indices],
                                    source=vals_left[valid_indices])

            valid_indices = (tis + 1) < num_bins
            valid_indices &= tis >= 0

            voxel_grid.index_add_(dim=0,
                                    index=xs[valid_indices]*num_bins + ys[valid_indices] * width *num_bins
                                            + (tis_long[valid_indices] + 1),
                                    source=vals_right[valid_indices])

            voxel_grid = voxel_grid.view(height, width,num_bins)

        return voxel_grid

    def norm(self,events):
        with torch.no_grad():
            nonzero_ev = (events != 0)
            num_nonzeros = nonzero_ev.sum()
            if num_nonzeros > 0: ## 실용 데이터가 있으면
                mean = events.sum() / num_nonzeros
                stddev = torch.sqrt((events ** 2).sum() / num_nonzeros - mean ** 2)
                mask = nonzero_ev.float()
                events = mask * (events - mean) / (stddev + 1e-8)

        return events

    def get_input_info_event(self,event_path, width, height):

        evs_stream=torch.zeros(0,4).float()

        evs_stream=np.load(event_path, allow_pickle=True)
        evs_stream=torch.from_numpy(evs_stream).float()
        evs_stream[:,0]=torch.from_numpy(self.fix_time(evs_stream[:,0].numpy()))

        ev_ten = self.ev2grid(evs_stream,width=width, height=height)
        
        if self.norm_e:
            events=self.norm(ev_ten)
        
        return events

    def get_input_info_event_binlast(self,event_path, width, height):

        evs_stream=torch.zeros(0,4).float()

        evs_stream=np.load(event_path, allow_pickle=True)
        evs_stream=torch.from_numpy(evs_stream).float()
        evs_stream[:,0]=torch.from_numpy(self.fix_time(evs_stream[:,0].numpy()))

        ev_ten = self.ev2grid_binlast(evs_stream,width=width, height=height)
        
        if self.norm_e:
            events=self.norm(ev_ten)
        
        return events