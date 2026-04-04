"""Shared mock data for oncall eval cases.

Mirrors the test fixtures from tests/test_schema_locator.py so eval cases
reference realistic schema structures.
"""

import json

SAMPLE_SCHEMA = {
    "type": "object",
    "properties": {
        "basicInfo": {
            "type": "void",
            "x-component": "CommonGroupItem",
            "x-component-props": {"label": "基础信息"},
            "properties": {
                "categoryId": {
                    "title": "商品品类",
                    "type": "number",
                    "x-component": "CategorySelect",
                    "attr_keys_scope": {"bind_attr_keys": ["category_id"]},
                    "reaction_rules": [],
                },
            },
        },
        "merchantInfo": {
            "type": "void",
            "x-component": "CommonGroupItem",
            "x-component-props": {"label": "商家信息"},
            "properties": {
                "accountName": {
                    "title": "商家名称",
                    "type": "string",
                    "x-component": "AccountName",
                    "attr_keys_scope": {"bind_attr_keys": ["account_name"]},
                    "required": True,
                    "reaction_rules": [{"actionTypeEnum": "VISIBLE"}],
                },
                "platformProductId": {
                    "title": "商家平台商品ID",
                    "type": "string",
                    "x-component": "PlatformProductId",
                    "attr_keys_scope": {"bind_attr_keys": ["platform_product_id"]},
                    "reaction_rules": [{"actionTypeEnum": "VISIBLE"}],
                },
            },
        },
    },
}

SAMPLE_DIM = {
    "category_full_name": "购物>果蔬生鲜>水果",
    "category_id": "5019003",
    "product_type": "1",
    "product_sub_type": "0",
    "template_type": "1",
    "template_sub_type": "0",
}

SAMPLE_DETAIL = {
    "online_template": [
        {
            "template_name": "test",
            "attr_list": [],
            "schema_config": json.dumps(SAMPLE_SCHEMA),
        }
    ]
}

PARENT_DIM = {
    **SAMPLE_DIM,
    "category_full_name": "购物>果蔬生鲜",
    "category_id": "5019000",
}
