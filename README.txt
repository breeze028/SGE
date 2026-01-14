This sample code is a Unity project that allows running the triangle soup => pineapple optimization example shown throughout our I3D 2024 presentation. Open the "OptimizerScene" scene to try it out. This Unity project should work with basically any Unity version, no need to download a specific one.



CONTROLS
- The target 3D model to optimize can be swapped out for another with the "target3Dmesh" variable. Using the "init primitives on mesh surface", the triangle soup will be initialized randomly along the target's surface, otherwise they are initialized randomly within the AABB of the target mesh.

- Press F1/F2 to switch between viewing the optimization or the target.

- Press P to pause. While paused, press Space to perform one individual optimization step.

- Disable the "Separate Free View Camera" option to see the actual optimization points of views rendered instead of the orbit camera.

- The "optimize colors separately" will make it so that each optimization step is performed twice, once only mutating the positions, once only mutating the colors. This trick allows better optimization convergence in this example by separating the gradient estimation of positions and colors.

- The "Primitive Resampling" is a trick used to make this use case with a soup of triangles much more efficient, by re-using invalid triangles to subdive valid ones in two using longest edge bisection. Without this technique, triangles falling outside the surface of the target mesh will simply become as small as they can without ever being useful. With this technique, they can instead be retargetted towards improving the quality of the surface's reconstruction. Triangles are deemed invalid when their area is under the specified threshold, or when they haven't been seen by any point of view for at least the specific amount of optimization steps.



IMPLEMENTATION DETAILS
- Note that, as described in the paper, we choose to use here an antithetic gradient estimator: for each random perturbation of the entire scene ("plus epsilon"), we also do the opposite perturbation ("minus epsilon"), and estimate the loss reduction based on the difference between these two results, instead of comparing between "perturbed" and "non-perturbed" results.

- All the compute shaders required by the method are in the "StochasticOptimizer.compute" file. Note that in the RandomPerturbation kernel, for simplicity, we use the learning rate as a perturbation amplitude parameter but should be a separate one, ideally the smallest possible perturbation that is guaranteed to produce a change in the output.

- The GradientEstimation compute shader is split in two kernels. "GradientEstimation" first runs over all the pixels in the images to evaluate per-pixel loss reductions. It accumulates this into a temporary buffer with one value per triangle (quantized in integer since float atomics are not yet available in Unity). "GradientEstimationPost" then runs with one thread per triangle to get this value and accumulates "mutation*loss_reduction" individually for each parameter of the triangle. This allows performing much less work during the per-pixel kernel, offloading the work that can be to a per-primitive kernel instead.