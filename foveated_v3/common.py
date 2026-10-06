"""Frozen model and provenance contract for foveated_v3 downstream jobs."""
import os
if not os.environ.get('SLURM_JOB_ID'):
    raise RuntimeError('Load numerical model code only inside Slurm')

import hashlib
import json
from pathlib import Path
import torch
from torch import nn
import train_mean as T

P = T.P


def identity_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


class VarianceHead(nn.Module):
    """Independent variance head; never modifies the fixed mean predictor."""
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(nn.Conv2d(772, 128, 1), nn.GELU(),
                                  nn.Conv2d(128, 128, 3, padding=1), nn.GELU(),
                                  nn.Conv2d(128, 128, 3, padding=2, dilation=2), nn.GELU())
        self.out = nn.Conv2d(128, 384, 1)
        nn.init.zeros_(self.out.weight)
        nn.init.zeros_(self.out.bias)

    def forward(self, partial, mean, resolution, weight):
        b, _, h, w = partial.shape
        yy, xx = torch.meshgrid(torch.linspace(-1, 1, h, device=partial.device),
                                torch.linspace(-1, 1, w, device=partial.device), indexing='ij')
        xy = torch.stack((xx, yy))[None].expand(b, -1, -1, -1)
        values = torch.cat((partial.detach(), mean.detach(), resolution, weight, xy), 1)
        return self.out(self.body(values)).clamp(-6, 4)


class FrozenObserver:
    def __init__(self, mean_run, variance_run=None, require_calibration=True):
        self.mean_run = Path(mean_run).resolve()
        self.config = json.loads((self.mean_run / 'config.json').read_text())
        self.frozen = json.loads((self.mean_run / 'frozen_mean.json').read_text())
        assert self.config['state'] in ('early_stopped', 'budget_exhausted')
        assert self.frozen['stage'] == 'mean_only_frozen'
        frozen_fields = {k:v for k,v in self.frozen.items() if k != 'mean_identity_hash'}
        self.mean_identity_hash = identity_hash(frozen_fields)
        assert self.mean_identity_hash == self.frozen['mean_identity_hash']
        self.mean_hash = P.digest(self.mean_run / 'best.pt')
        assert self.mean_hash == self.frozen['best_checkpoint_sha256'] == self.config['best_checkpoint_sha256']
        self.protocol = json.loads((self.mean_run / 'protocol.json').read_text())
        self.protocol_hash = P.digest(self.mean_run / 'protocol.json')
        assert self.protocol_hash == self.frozen['protocol_sha256'] == self.config['protocol_sha256']
        self.options = self.protocol['observer_kwargs']
        self.maximum_budget = int(self.protocol['maximum_budget'])
        assert self.maximum_budget == self.config['maximum_budget'] == self.frozen['maximum_budget']
        source_dir = Path(T.__file__).parent
        self.observer_hash = P.digest(source_dir / 'observer.py')
        assert self.observer_hash == self.frozen['observer_sha256'] == self.protocol['source_sha256']['observer.py']
        self.bound_files = {self.mean_run/'best.pt':self.mean_hash,
                            self.mean_run/'protocol.json':self.protocol_hash,
                            self.mean_run/'normalization.pt':self.frozen['normalization_sha256']}
        # Reuse precisely the model implementation used to produce the checkpoint.
        for filename in ('train_mean.py', 'observer.py', 'frozen_mean.py',
                         'frozen_data.py', 'frozen_legacy_observer.py'):
            expected = self.frozen['source_sha256'][filename]
            assert P.digest(source_dir / filename) == expected, filename
            self.bound_files[source_dir / filename] = expected
        assert P.digest(T.MANIFEST) == self.config['manifest_sha256']
        self.manifest = json.loads(T.MANIFEST.read_text())
        assert P.digest(self.manifest['raw_fixations']) == self.config['fixations_sha256']
        self.bound_files[T.MANIFEST] = self.config['manifest_sha256']
        self.bound_files[Path(self.manifest['raw_fixations'])] = self.config['fixations_sha256']
        checkpoint = torch.load(self.mean_run / 'best.pt', map_location='cpu', weights_only=False)
        assert checkpoint['epoch'] == self.frozen['best_epoch']
        assert checkpoint['dev_mse'] == self.frozen['best_dev_mse']
        self.normalization = torch.load(self.mean_run / 'normalization.pt', weights_only=True)
        assert P.digest(self.mean_run / 'normalization.pt') == self.frozen['normalization_sha256']
        for key in ('mean', 'std'):
            assert torch.equal(checkpoint['normalization'][key], self.normalization[key])
        self.feature_mean = self.normalization['mean'].cuda()[None, :, None, None]
        self.feature_std = self.normalization['std'].cuda()[None, :, None, None]
        assert (self.feature_std > 0).all()
        self.encoder = T.M.load_frozen('cuda').eval().requires_grad_(False)
        self.mean_model = T.M.DirectMean().cuda().eval().requires_grad_(False)
        self.mean_model.load_state_dict(checkpoint['model'])
        self.target_cache = Path(self.config['target_cache'])
        self.development_cache = Path(self.config['development_cache'])
        self.cache_index = json.loads((self.mean_run / 'cache_index.json').read_text())
        self._checked_targets = set()
        self.variance = None
        self.variance_hash = None
        self.common_hash = P.digest(__file__)
        self.bound_files[Path(__file__)] = self.common_hash
        self.identity = {'version':'foveated_v3', 'mean_identity_hash':self.mean_identity_hash,
                         'mean_checkpoint_sha256':self.mean_hash, 'protocol_sha256':self.protocol_hash,
                         'observer_sha256':self.observer_hash, 'maximum_budget':self.maximum_budget,
                         'normalization_sha256':self.frozen['normalization_sha256'],
                         'common_sha256':self.common_hash, 'forward_batch_size':1}
        if variance_run is not None:
            variance_run = Path(variance_run).resolve()
            self.variance_config = json.loads((variance_run / 'config.json').read_text())
            assert self.variance_config['state'] in ('calibration_review_required', 'completed')
            assert self.variance_config['mean_identity_hash'] == self.mean_identity_hash
            assert self.variance_config['common_sha256'] == self.common_hash
            self.variance_hash = P.digest(variance_run / 'best.pt')
            assert self.variance_hash == self.variance_config['variance_checkpoint_sha256']
            value = torch.load(variance_run / 'best.pt', map_location='cpu', weights_only=False)
            assert value['mean_identity_hash'] == self.mean_identity_hash
            self.variance = VarianceHead().cuda().eval().requires_grad_(False)
            self.variance.load_state_dict(value['model'])
            self.bound_files[variance_run/'best.pt'] = self.variance_hash
            self.identity['variance_checkpoint_sha256'] = self.variance_hash
            if require_calibration:
                review_path = variance_run / 'calibration_review.json'
                review = json.loads(review_path.read_text())
                assert review['state'] == 'accepted'
                assert review['mean_identity_hash'] == self.mean_identity_hash
                assert review['variance_checkpoint_sha256'] == self.variance_hash
                summary_path = variance_run / 'calibration_summary.json'
                assert review['calibration_summary_sha256'] == P.digest(summary_path)
                self.bound_files[review_path] = P.digest(review_path)
                self.bound_files[summary_path] = P.digest(summary_path)
                self.identity['calibration_review_sha256'] = P.digest(review_path)
        self.identity_hash = identity_hash(self.identity)

    @torch.no_grad()
    def mean_current(self, rgb, resolution, weight):
        partial = (self.encoder(rgb) - self.feature_mean) / self.feature_std
        mean = self.mean_model(partial, resolution.cuda(), weight.cuda())
        assert torch.isfinite(mean).all()
        return partial, mean

    @torch.no_grad()
    def current(self, rgb, resolution, weight):
        assert len(rgb) == 1, 'Canonical evaluation must encode each observation individually'
        if self.variance is None:
            raise RuntimeError('Variance checkpoint required for U/G evaluation')
        partial, mean = self.mean_current(rgb, resolution, weight)
        logvar = self.variance(partial, mean, resolution.cuda(), weight.cuda())
        assert torch.isfinite(logvar).all()
        return partial, mean, logvar

    def target_for_labels(self, name):
        if name not in self._checked_targets:
            assert P.digest(self.target_cache / (name+'.pt')) == self.cache_index['target_sha256'][name]
            self._checked_targets.add(name)
        value = torch.load(self.target_cache / (name+'.pt'), weights_only=True)
        return (value['features'][None].cuda().float() - self.feature_mean) / self.feature_std

    def assert_unchanged(self):
        for path, expected in self.bound_files.items():
            assert P.digest(path) == expected, str(path)
        for model in (self.encoder, self.mean_model):
            assert not model.training and all(not p.requires_grad for p in model.parameters())
        # Check in-memory mean too: file hashes alone cannot detect optimizer mutation.
        expected = torch.load(self.mean_run/'best.pt', map_location='cpu', weights_only=False)['model']
        assert all(torch.equal(value.detach().cpu(), expected[key])
                   for key,value in self.mean_model.state_dict().items())
