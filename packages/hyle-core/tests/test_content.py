from hyle.content import Binary, ResourceRef


def test_binary_owns_mutable_input() -> None:
    source = bytearray(b"abc")
    value = Binary(source, media_type="image/png", name="image.png")
    source[:] = b"xyz"
    assert value.data == b"abc"
    assert value.size == 3


def test_resource_ref_is_reference_only() -> None:
    value = ResourceRef("https://example.com/image.png", media_type="image/png")
    assert value.uri == "https://example.com/image.png"
    assert value.media_type == "image/png"
