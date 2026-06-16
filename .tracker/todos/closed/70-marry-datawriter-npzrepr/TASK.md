# Marry DataWriter with NpzRepresentation (IORepresentationMixin) somehow

**Created**: 2024-11-10
**Closed**: 2024-11-10
**Priority**: 3

## Description

DataWriter should most likely just call the representation's abstract methods but it's unclear what to do with the current behavior (binary_format/image_format/compress etc.). We need a way to express the output format dynamically.
