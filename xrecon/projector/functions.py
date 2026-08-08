from torch.autograd import Function

_C = None
_IMPORT_ERROR = None


def _backend():
    global _C, _IMPORT_ERROR
    if _C is not None:
        return _C
    try:
        import xrecon_C as backend
    except Exception as exc:
        _IMPORT_ERROR = exc
        raise ImportError("Failed to import 'xrecon_C' native extension. Build/install XRecon first.") from exc
    _C = backend
    return _C


def _projection_transpose2d(input, option, angles):
    return _backend().projection_transpose2d(input, option, angles)


def _backprojection2d(input, option, angles):
    return _backend().backward_projection2d(input, option, angles)


def _weighted_backprojection2d(input, option, angles):
    return _backend().weighted_backward_projection2d(input, option, angles)


def _backprojection_transpose2d(input, option, angles):
    return _backend().backward_projection_transpose2d(input, option, angles)


def _weighted_backprojection_transpose2d(input, option, angles):
    return _backend().weighted_backward_projection_transpose2d(input, option, angles)


def system_matrix2d(option, angles):
    return _backend().system_matrix2d(option, angles)


def system_matrix3d(option, angles):
    return _backend().system_matrix3d(option, angles)


def _projection3d(input, option, angles):
    return _backend().forward_projection3d(input, option, angles)


def _projection_transpose3d(input, option, angles):
    return _backend().projection_transpose3d(input, option, angles)


def _backprojection3d(input, option, angles):
    return _backend().backward_projection3d(input, option, angles)


def _weighted_backprojection3d(input, option, angles):
    return _backend().weighted_backward_projection3d(input, option, angles)


def _backprojection_transpose3d(input, option, angles):
    return _backend().backward_projection_transpose3d(input, option, angles)


def _weighted_backprojection_transpose3d(input, option, angles):
    return _backend().weighted_backward_projection_transpose3d(input, option, angles)


def _projection_helical3d(input, option, angles, source_pos):
    return _backend().forward_projection_helical3d(input, option, angles, source_pos)


def _projection_transpose_helical3d(input, option, angles, source_pos):
    return _backend().projection_transpose_helical3d(input, option, angles, source_pos)


def _backprojection_helical3d(input, option, angles, source_pos):
    return _backend().backward_projection_helical3d(input, option, angles, source_pos)


def _weighted_backprojection_helical3d(input, option, angles, source_pos):
    return _backend().weighted_backward_projection_helical3d(input, option, angles, source_pos)


def _backprojection_transpose_helical3d(input, option, angles, source_pos):
    return _backend().backward_projection_transpose_helical3d(input, option, angles, source_pos)


def _weighted_backprojection_transpose_helical3d(input, option, angles, source_pos):
    return _backend().weighted_backward_projection_transpose_helical3d(input, option, angles, source_pos)


class Projection2d(Function):
    @staticmethod    
    def forward(ctx, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = _backend().forward_projection2d(input, option, angles)
        ctx.save_for_backward(option, angles)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        option, angles = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = _projection_transpose2d(grad_output, option, angles)
        return grad_input, None, None, None


class BackProjection2d(Function):
    @staticmethod
    def forward(ctx, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = _backprojection2d(input, option, angles)
        ctx.save_for_backward(option, angles)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        option, angles = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = _backprojection_transpose2d(grad_output, option, angles)
        return grad_input, None, None, None

class WeightedBackProjection2d(Function):
    @staticmethod
    def forward(ctx, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = _weighted_backprojection2d(input, option, angles)
        ctx.save_for_backward(option, angles)
        return output
    
    @staticmethod
    def backward(ctx, grad_output):
        option, angles = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = _weighted_backprojection_transpose2d(grad_output, option, angles)
        return grad_input, None, None, None


class ProjectionTranspose2d(Function):
    @staticmethod
    def forward(ctx, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = _projection_transpose2d(input, option, angles)
        ctx.save_for_backward(option, angles)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        option, angles = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = _backend().forward_projection2d(grad_output, option, angles)
        return grad_input, None, None, None


class BackProjectionTranspose2d(Function):
    @staticmethod
    def forward(ctx, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = _backprojection_transpose2d(input, option, angles)
        ctx.save_for_backward(option, angles)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        option, angles = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = _backprojection2d(grad_output, option, angles)
        return grad_input, None, None, None


class WeightedBackProjectionTranspose2d(Function):
    @staticmethod
    def forward(ctx, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = _weighted_backprojection_transpose2d(input, option, angles)
        ctx.save_for_backward(option, angles)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        option, angles = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        grad_input = _weighted_backprojection2d(grad_output, option, angles)
        return grad_input, None, None, None


class Projection3d(Function):
    @staticmethod
    def forward(ctx, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = _projection3d(input, option, angles)
        ctx.save_for_backward(option, angles)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        option, angles = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        return _projection_transpose3d(grad_output, option, angles), None, None, None


class ProjectionTranspose3d(Function):
    @staticmethod
    def forward(ctx, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = _projection_transpose3d(input, option, angles)
        ctx.save_for_backward(option, angles)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        option, angles = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        return _projection3d(grad_output, option, angles), None, None, None


class BackProjection3d(Function):
    @staticmethod
    def forward(ctx, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = _backprojection3d(input, option, angles)
        ctx.save_for_backward(option, angles)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        option, angles = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        return _backprojection_transpose3d(grad_output, option, angles), None, None, None


class BackProjectionTranspose3d(Function):
    @staticmethod
    def forward(ctx, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = _backprojection_transpose3d(input, option, angles)
        ctx.save_for_backward(option, angles)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        option, angles = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        return _backprojection3d(grad_output, option, angles), None, None, None


class WeightedBackProjection3d(Function):
    @staticmethod
    def forward(ctx, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = _weighted_backprojection3d(input, option, angles)
        ctx.save_for_backward(option, angles)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        option, angles = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        return _weighted_backprojection_transpose3d(grad_output, option, angles), None, None, None


class WeightedBackProjectionTranspose3d(Function):
    @staticmethod
    def forward(ctx, input, option, angles):
        if not input.is_contiguous():
            input = input.contiguous()
        output = _weighted_backprojection_transpose3d(input, option, angles)
        ctx.save_for_backward(option, angles)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        option, angles = ctx.saved_tensors
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()
        return _weighted_backprojection3d(grad_output, option, angles), None, None, None


class ProjectionHelical3d(Function):
    @staticmethod
    def forward(ctx, input, option, angles, source_pos):
        input = input.contiguous()
        source_pos = source_pos.contiguous()
        output = _projection_helical3d(input, option, angles, source_pos)
        ctx.save_for_backward(option, angles, source_pos)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        option, angles, source_pos = ctx.saved_tensors
        return _projection_transpose_helical3d(grad_output.contiguous(), option, angles, source_pos), None, None, None


class ProjectionTransposeHelical3d(Function):
    @staticmethod
    def forward(ctx, input, option, angles, source_pos):
        source_pos = source_pos.contiguous()
        output = _projection_transpose_helical3d(input.contiguous(), option, angles, source_pos)
        ctx.save_for_backward(option, angles, source_pos)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        option, angles, source_pos = ctx.saved_tensors
        return _projection_helical3d(grad_output.contiguous(), option, angles, source_pos), None, None, None


class BackProjectionHelical3d(Function):
    @staticmethod
    def forward(ctx, input, option, angles, source_pos):
        source_pos = source_pos.contiguous()
        output = _backprojection_helical3d(input.contiguous(), option, angles, source_pos)
        ctx.save_for_backward(option, angles, source_pos)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        option, angles, source_pos = ctx.saved_tensors
        return _backprojection_transpose_helical3d(grad_output.contiguous(), option, angles, source_pos), None, None, None


class BackProjectionTransposeHelical3d(Function):
    @staticmethod
    def forward(ctx, input, option, angles, source_pos):
        source_pos = source_pos.contiguous()
        output = _backprojection_transpose_helical3d(input.contiguous(), option, angles, source_pos)
        ctx.save_for_backward(option, angles, source_pos)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        option, angles, source_pos = ctx.saved_tensors
        return _backprojection_helical3d(grad_output.contiguous(), option, angles, source_pos), None, None, None


class WeightedBackProjectionHelical3d(Function):
    @staticmethod
    def forward(ctx, input, option, angles, source_pos):
        source_pos = source_pos.contiguous()
        output = _weighted_backprojection_helical3d(input.contiguous(), option, angles, source_pos)
        ctx.save_for_backward(option, angles, source_pos)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        option, angles, source_pos = ctx.saved_tensors
        return _weighted_backprojection_transpose_helical3d(grad_output.contiguous(), option, angles, source_pos), None, None, None


class WeightedBackProjectionTransposeHelical3d(Function):
    @staticmethod
    def forward(ctx, input, option, angles, source_pos):
        source_pos = source_pos.contiguous()
        output = _weighted_backprojection_transpose_helical3d(input.contiguous(), option, angles, source_pos)
        ctx.save_for_backward(option, angles, source_pos)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        option, angles, source_pos = ctx.saved_tensors
        return _weighted_backprojection_helical3d(grad_output.contiguous(), option, angles, source_pos), None, None, None
