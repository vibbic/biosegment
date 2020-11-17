<template>
  <v-form v-model="valid" ref="form" lazy-validation>
    <v-text-field label="Title" v-model="dataset.title" required></v-text-field>
    <v-text-field
      label="Description"
      v-model="dataset.description"
    ></v-text-field>
    <v-select
      label="File type"
      v-model="dataset.file_type"
      :items="fileTypes"
      deletable-chips
      chips
    ></v-select>
    <v-text-field label="Location" v-model="dataset.location"></v-text-field>
    <v-slider
      label="Resolution X"
      v-model="dataset.resolution.x"
      min="1"
      max="5000"
    >
      <template v-slot:append>
        <v-text-field
          v-model="dataset.resolution.x"
          class="mt-0 pt-0"
          type="number"
          style="width: 60px"
        ></v-text-field>
      </template>
    </v-slider>
    <v-slider
      label="Resolution Y"
      v-model="dataset.resolution.y"
      min="1"
      max="5000"
    >
      <template v-slot:append>
        <v-text-field
          v-model="dataset.resolution.y"
          class="mt-0 pt-0"
          type="number"
          style="width: 60px"
        ></v-text-field>
      </template>
    </v-slider>
    <v-slider
      label="Resolution Z"
      v-model="dataset.resolution.z"
      min="1"
      max="5000"
    >
      <template v-slot:append>
        <v-text-field
          v-model="dataset.resolution.z"
          class="mt-0 pt-0"
          type="number"
          style="width: 60px"
        ></v-text-field>
      </template>
    </v-slider>
    <v-text-field
      label="Modality"
      v-model="dataset.modality"
      disabled
    ></v-text-field>
  </v-form>
</template>

<script lang="ts">
import { Component, Model, Vue } from 'vue-property-decorator';
import { DatasetCreate, DatasetFileType } from '@/api';
import { defaultDataset } from '@/interfaces';

@Component
export default class DatasetForm extends Vue {
  @Model('dataset', { type: DatasetCreate }) public dataset!: DatasetCreate;
  public fileTypes = Object.values(DatasetFileType);
  public valid = false;
}
</script>
