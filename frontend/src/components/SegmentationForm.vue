<template>
  <v-form v-model="valid" ref="form" lazy-validation>
    <v-text-field label="Title" v-model="segmentation.title" required></v-text-field>
    <v-text-field
      label="Description"
      v-model="segmentation.description"
    ></v-text-field>
    <v-select
      label="File type"
      v-model="segmentation.file_type"
      :items="file_types"
      deletable-chips
      chips
    ></v-select>
    <v-text-field label="Location" v-model="segmentation.location"></v-text-field>
  </v-form>
</template>

<script lang="ts">
import { Component, Model, Vue } from 'vue-property-decorator';
import { SegmentationCreate, SegmentationFileType } from '@/api';
import { defaultSegmentation } from '@/interfaces';

@Component
export default class SegmentationForm extends Vue {
  @Model('segmentation', { type: SegmentationCreate }) public segmentation!: SegmentationCreate;
  public file_types = Object.values(SegmentationFileType);
  public valid = false;
}
</script>
