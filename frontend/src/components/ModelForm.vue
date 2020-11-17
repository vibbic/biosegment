<template>
  <v-form v-model="valid" ref="form" lazy-validation>
    <v-text-field label="Title" v-model="model.title" required></v-text-field>
    <v-text-field
      label="Description"
      v-model="model.description"
    ></v-text-field>
    <v-text-field label="Location" v-model="model.location"></v-text-field>
  </v-form>
</template>

<script lang="ts">
import { Component, Model, Vue } from 'vue-property-decorator';
import { ModelCreate } from '@/api';
import { defaultModel } from '@/interfaces';

@Component
export default class ModelForm extends Vue {
  @Model('model', { type: ModelCreate }) public model!: ModelCreate;
  public valid = false;
}
</script>
