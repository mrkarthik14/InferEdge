"""Deterministic synthetic work. This is deliberately NOT a language model."""


class MockTokenizer:
    bos_token_id = 1
    eos_token_id = 2
    pad_token_id = 0

    def encode(self, text: str) -> list[int]:
        return [self.bos_token_id, *(byte + 3 for byte in text.encode("utf-8"))]

    def decode(self, token_ids: list[int]) -> str:
        return bytes(token - 3 for token in token_ids if token >= 3).decode(
            "utf-8", errors="replace"
        )


class MockModel:
    def generate(self, input_ids: list[int], max_new_tokens: int) -> list[int]:
        # Recompute a stable prefix hash at every step, with no mutable cache.
        # Output is printable ASCII, so a generated token is one output byte.
        output = list(input_ids)
        for _ in range(max_new_tokens):
            state = 2166136261
            for token in output:
                state = ((state ^ token) * 16777619) & 0xFFFFFFFF
            output.append(32 + state % 95 + 3)
        return output
