from __future__ import annotations

import json

import q1_predict
import q2_optimize
import q3_dynamic
import q4_compare
import validate_outputs


def run():
    return {
        "q1": q1_predict.run(),
        "q2": q2_optimize.run(),
        "q3": q3_dynamic.run(),
        "q4": q4_compare.run(),
        "validation": validate_outputs.run(),
    }


if __name__ == "__main__":
    print(json.dumps(run(), ensure_ascii=False, indent=2, default=str))

