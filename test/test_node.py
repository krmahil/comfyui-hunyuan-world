"""
Basic tests for ComfyUI HunyuanWorld node registration and structure.
These tests verify node loading without requiring the full hy3dworld installation.
"""
import pytest
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))


class TestNodeRegistration:
    """Test that nodes register correctly with ComfyUI."""

    def test_node_class_mappings_exist(self):
        from hunyuan_world_node import NODE_CLASS_MAPPINGS
        assert isinstance(NODE_CLASS_MAPPINGS, dict)
        assert len(NODE_CLASS_MAPPINGS) > 0

    def test_node_display_name_mappings_exist(self):
        from hunyuan_world_node import NODE_DISPLAY_NAME_MAPPINGS
        assert isinstance(NODE_DISPLAY_NAME_MAPPINGS, dict)
        assert len(NODE_DISPLAY_NAME_MAPPINGS) > 0

    def test_all_nodes_have_display_names(self):
        from hunyuan_world_node import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
        for key in NODE_CLASS_MAPPINGS:
            assert key in NODE_DISPLAY_NAME_MAPPINGS, f"Missing display name for {key}"

    def test_expected_nodes_registered(self):
        from hunyuan_world_node import NODE_CLASS_MAPPINGS
        expected = [
            "HYWorldText2Pano",
            "HYWorldImage2Pano",
            "HYWorldSceneGen",
            "HYWorldFullPipeline",
            "HYWorldUnloadModels",
        ]
        for name in expected:
            assert name in NODE_CLASS_MAPPINGS, f"Missing node: {name}"


class TestNodeStructure:
    """Test that each node has required ComfyUI attributes."""

    def _get_node_classes(self):
        from hunyuan_world_node import NODE_CLASS_MAPPINGS
        return NODE_CLASS_MAPPINGS

    def test_all_nodes_have_input_types(self):
        for name, cls in self._get_node_classes().items():
            assert hasattr(cls, "INPUT_TYPES"), f"{name} missing INPUT_TYPES"
            assert callable(cls.INPUT_TYPES), f"{name}.INPUT_TYPES not callable"
            inputs = cls.INPUT_TYPES()
            assert "required" in inputs or "optional" in inputs, \
                f"{name}.INPUT_TYPES must have 'required' or 'optional'"

    def test_all_nodes_have_return_types(self):
        for name, cls in self._get_node_classes().items():
            assert hasattr(cls, "RETURN_TYPES"), f"{name} missing RETURN_TYPES"
            assert isinstance(cls.RETURN_TYPES, tuple), f"{name}.RETURN_TYPES not tuple"

    def test_all_nodes_have_function(self):
        for name, cls in self._get_node_classes().items():
            assert hasattr(cls, "FUNCTION"), f"{name} missing FUNCTION"
            func_name = cls.FUNCTION
            assert hasattr(cls, func_name), f"{name} missing method '{func_name}'"

    def test_all_nodes_have_category(self):
        for name, cls in self._get_node_classes().items():
            assert hasattr(cls, "CATEGORY"), f"{name} missing CATEGORY"
            assert cls.CATEGORY.startswith("3d/"), f"{name} category should start with '3d/'"

    def test_text2pano_inputs(self):
        nodes = self._get_node_classes()
        inputs = nodes["HYWorldText2Pano"].INPUT_TYPES()
        assert "prompt" in inputs["required"]

    def test_image2pano_inputs(self):
        nodes = self._get_node_classes()
        inputs = nodes["HYWorldImage2Pano"].INPUT_TYPES()
        assert "image" in inputs["required"]

    def test_scenegen_inputs(self):
        nodes = self._get_node_classes()
        inputs = nodes["HYWorldSceneGen"].INPUT_TYPES()
        assert "panorama_path" in inputs["required"]


class TestModuleInit:
    """Test __init__.py module imports."""

    def test_init_imports(self):
        # Test that __init__.py properly re-exports mappings
        import importlib
        spec = importlib.util.spec_from_file_location(
            "comfyui_hunyuan_world",
            os.path.join(os.path.dirname(os.path.dirname(__file__)), "__init__.py")
        )
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        assert hasattr(mod, "NODE_CLASS_MAPPINGS")
        assert hasattr(mod, "NODE_DISPLAY_NAME_MAPPINGS")
        assert hasattr(mod, "__version__")
