from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import ray
from ray.rllib.optimizers.sync_samples_optimizer import SyncSamplesOptimizer
from ray.rllib.evaluation.sample_batch import SampleBatch
from ray.rllib.utils.filter import RunningStat
from ray.rllib.utils.timer import TimerStat
import numpy as np

class TRPOSyncSamplesOptimizer(SyncSamplesOptimizer):
    """A simple synchronous RL optimizer.

    In each step, this optimizer pulls samples from a number of remote
    evaluators, concatenates them, and then updates a local model. The updated
    model weights are then broadcast to all remote evaluators.
    """

    def _init(self, num_sgd_iter=1, train_batch_size=1):
        self.update_weights_timer = TimerStat()
        self.sample_timer = TimerStat()
        self.grad_timer = TimerStat()
        self.throughput = RunningStat()
        self.num_sgd_iter = num_sgd_iter
        self.train_batch_size = train_batch_size
        self.learner_stats = {}

    def step(self):
        with self.update_weights_timer:
            if self.remote_evaluators:
                weights = ray.put(self.local_evaluator.get_weights())
                for e in self.remote_evaluators:
                    e.set_weights.remote(weights)

        with self.sample_timer:
            samples = []
            while sum(s.count for s in samples) < self.train_batch_size:
                if self.remote_evaluators:
                    samples.extend(
                        ray.get([
                            e.sample.remote() for e in self.remote_evaluators
                        ]))
                else:
                    samples.append(self.local_evaluator.sample())
            samples = SampleBatch.concat_samples(samples)
            self.sample_timer.push_units_processed(samples.count)

        with self.grad_timer:
            final_fetches = None
            for i in range(self.num_sgd_iter):
                self.local_evaluator.curr_lr = 1.0
                weights = ray.put(self.local_evaluator.get_weights())
                for _ in range(10):
                    self.local_evaluator.set_weights(weights)
                    gradients = self.local_evaluator.compute_gradients(samples)
                    fetches = self.local_evaluator.apply_gradients(gradients)
                    final_fetches = fetches
                    kl = fetches["kl"]
                    if self.local_evaluator.should_update_kl(kl):
                        self.local_evaluator.curr_lr = self.local_evaluator.curr_lr * 0.5
                if "stats" in fetches:
                    self.learner_stats = fetches["stats"]
                if self.num_sgd_iter > 1:
                    print(i, fetches)
            self.grad_timer.push_units_processed(samples.count)

        self.num_steps_sampled += samples.count
        self.num_steps_trained += samples.count
        return final_fetches

    def stats(self):
        return dict(
            SyncSamplesOptimizer.stats(self), **{
                "sample_time_ms": round(1000 * self.sample_timer.mean, 3),
                "grad_time_ms": round(1000 * self.grad_timer.mean, 3),
                "update_time_ms": round(1000 * self.update_weights_timer.mean,
                                        3),
                "opt_peak_throughput": round(self.grad_timer.mean_throughput,
                                             3),
                "sample_peak_throughput": round(
                    self.sample_timer.mean_throughput, 3),
                "opt_samples": round(self.grad_timer.mean_units_processed, 3),
                "learner": self.learner_stats,
            })

def cg(f_Ax, b, cg_iters=10, callback=None, verbose=False, residual_tol=1e-10):
    """
    Demmel p 312
    """
    p = b.copy()
    r = b.copy()
    x = np.zeros_like(b)
    rdotr = r.dot(r)

    fmtstr =  "%10i %10.3g %10.3g"
    titlestr =  "%10s %10s %10s"
    if verbose: print(titlestr % ("iter", "residual norm", "soln norm"))

    for i in range(cg_iters):
        if callback is not None:
            callback(x)
        if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = f_Ax(p)
        v = rdotr / p.dot(z)
        x += v*p
        r -= v*z
        newrdotr = r.dot(r)
        mu = newrdotr/rdotr
        p = r + mu*p

        rdotr = newrdotr
        if rdotr < residual_tol:
            break

    if callback is not None:
        callback(x)
    if verbose: print(fmtstr % (i+1, rdotr, np.linalg.norm(x)))  # pylint: disable=W0631
    return x
