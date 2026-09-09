"""The Gunkel reference payload this package publishes alongside its partitions."""

from __future__ import annotations

from tehillim_cluster.export_gunkel import build_gunkel_payload


class TestBuildGunkelPayload:
    def test_it_names_every_genre(self) -> None:
        payload = build_gunkel_payload()

        assert payload["genres"]

    def test_it_maps_psalms_to_their_genres(self) -> None:
        payload = build_gunkel_payload()

        assert payload["psalms"]

    def test_it_names_every_family(self) -> None:
        assert build_gunkel_payload()["families"]

    def test_the_payload_is_json_serialisable(self) -> None:
        """It is written straight to disk, so an unserialisable value would fail at the end."""
        import json

        assert json.dumps(build_gunkel_payload())
