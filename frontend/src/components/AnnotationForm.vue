<template>
  <v-form v-model="valid" ref="form" lazy-validation>
    <v-text-field label="Title" v-model="annotation.title" required></v-text-field>
    <v-text-field
      label="Description"
      v-model="annotation.description"
    ></v-text-field>
    <v-select
      label="File type"
      v-model="annotation.file_type"
      :items="file_types"
      deletable-chips
      chips
    ></v-select>
    <v-text-field label="Location" v-model="annotation.location"></v-text-field>
  </v-form>
</template>

<script lang="ts">
import { Component, Model, Vue } from 'vue-property-decorator';
import { AnnotationCreate, AnnotationFileType } from '@/api';
import { defaultAnnotation } from '@/interfaces';

@Component
export default class AnnotationForm extends Vue {
  @Model('annotation') public annotation!: AnnotationCreate;
  public file_types = Object.values(AnnotationFileType);
  public valid = false;
}
</script>
