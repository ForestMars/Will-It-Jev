# -*- coding: utf-8 -*-
# Generated from proto/tier3.proto. Do not edit by hand.
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf.internal import builder as _builder

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0btier3.proto\x12\x05tier3"\x1e\n\x0cScoreRequest\x12\x0e\n\x06prompt\x18\x01 \x01(\t"W\n\rScoreResponse\x12\x11\n\traw_score\x18\x01 \x01(\x02\x12\x12\n\nconfidence\x18\x02 \x01(\x02\x12\r\n\x05label\x18\x03 \x01(\t\x12\x10\n\x08model_id\x18\x04 \x01(\t"j\n\tTaskLabel\x12\n\n\x02id\x18\x01 \x01(\t\x12\x0c\n\x04name\x18\x02 \x01(\t\x12\x13\n\x0bdescription\x18\x03 \x01(\t\x12\x1c\n\x14classifier_criterion\x18\x04 \x01(\t\x12\x10\n\x08group_id\x18\x05 \x01(\t"|\n\x08TaskSpec\x12\x0c\n\x04task\x18\x01 \x01(\t\x12&\n\x0boutput_kind\x18\x02 \x01(\x0e2\x11.tier3.OutputKind\x12 \n\x06labels\x18\x03 \x03(\x0b2\x10.tier3.TaskLabel\x12\x18\n\x10taxonomy_version\x18\x04 \x01(\t"W\n\x11ScoreTasksRequest\x12\x0e\n\x06prompt\x18\x01 \x01(\t\x12\r\n\x05tasks\x18\x02 \x03(\t\x12#\n\ntask_specs\x18\x03 \x03(\x0b2\x0f.tier3.TaskSpec"*\n\nLabelScore\x12\r\n\x05label\x18\x01 \x01(\t\x12\r\n\x05score\x18\x02 \x01(\x02"\x87\x01\n\tTaskScore\x12\x0c\n\x04task\x18\x01 \x01(\t\x12\r\n\x05label\x18\x02 \x01(\t\x12\x12\n\nconfidence\x18\x03 \x01(\x02\x12!\n\x06scores\x18\x04 \x03(\x0b2\x11.tier3.LabelScore\x12&\n\x0boutput_kind\x18\x05 \x01(\x0e2\x11.tier3.OutputKind"^\n\x12ScoreTasksResponse\x12\x10\n\x08model_id\x18\x01 \x01(\t\x12!\n\x07results\x18\x02 \x03(\x0b2\x10.tier3.TaskScore\x12\x13\n\x0bunsupported\x18\x03 \x03(\t*b\n\nOutputKind\x12\x1b\n\x17OUTPUT_KIND_UNSPECIFIED\x10\x00\x12\n\n\x06BINARY\x10\x01\x12\x0e\n\nMULTICLASS\x10\x02\x12\x0e\n\nMULTILABEL\x10\x03\x12\x0b\n\x07ORDINAL\x10\x042\x84\x01\n\x0bTier3Scorer\x122\n\x05Score\x12\x13.tier3.ScoreRequest\x1a\x14.tier3.ScoreResponse\x12A\n\nScoreTasks\x12\x18.tier3.ScoreTasksRequest\x1a\x19.tier3.ScoreTasksResponseb\x06proto3'
)

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'tier3_pb2', _globals)
